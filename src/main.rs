use std::{collections::HashMap, env, net::SocketAddr, sync::Arc};

use axum::{
    extract::{FromRef, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};

use anyhow::{anyhow, Context};
use cml_chain::{
    address::Address,
    assets::{AssetName, MultiAsset, Value as CmlValue},
    builders::output_builder::TransactionOutputBuilder,
    plutus::{
        CostModels, Language, PlutusData, PlutusV1Script, PlutusV2Script, PlutusV3Script,
    },
    transaction::{DatumOption, TransactionInput},
    PolicyId, Script,
};
use cml_core::serialization::{
    Deserialize as CmlDeserialize, RawBytesEncoding, Serialize as CmlSerialize,
};
use cml_crypto::{DatumHash, TransactionHash};
use blockfrost::BlockfrostAPI;
use blockfrost_openapi::models::epoch_param_content::EpochParamContent;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;
use tokio::signal;
use tracing::info;
use tracing_subscriber::EnvFilter;
use uplc::tx;

#[derive(Clone)]
struct AppState {
    initial_budget: (u64, u64),
    slot_config: (u64, u64, u32),
    run_phase_one: bool,
    cost_models: Arc<Vec<u8>>,
    blockfrost_api: Arc<BlockfrostAPI>,
}

impl FromRef<AppState> for Arc<BlockfrostAPI> {
    fn from_ref(app_state: &AppState) -> Arc<BlockfrostAPI> {
        Arc::clone(&app_state.blockfrost_api)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvaluationRequest {
    cbor: String,
    #[serde(default)]
    additional_utxo_set: Vec<(TxIn, TxOut)>,
}

#[derive(Debug, Deserialize)]
struct TxIn {
    #[serde(rename = "txId")]
    tx_id: String,
    index: u32,
}

#[derive(Debug, Deserialize)]
struct TxOut {
    address: String,
    value: ValueJson,
    #[serde(rename = "datumHash")]
    datum_hash: Option<String>,
    #[serde(default)]
    datum: Option<DatumField>,
    #[serde(default)]
    script: Option<ScriptJson>,
}

#[derive(Debug, Deserialize)]
struct ValueJson {
    coins: u64,
    assets: Option<HashMap<String, u64>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DatumField {
    Hex(String),
    Object(JsonValue),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ScriptJson {
    PlutusV1 {
        #[serde(rename = "plutus:v1")]
        hex: String,
    },
    PlutusV2 {
        #[serde(rename = "plutus:v2")]
        hex: String,
    },
    PlutusV3 {
        #[serde(rename = "plutus:v3")]
        hex: String,
    },
}

#[derive(Debug, Serialize)]
struct EvaluationResponse {
    redeemers: Vec<RedeemerEvaluation>,
}

#[derive(Debug, Serialize)]
struct RedeemerEvaluation {
    /// Hex encoded CBOR for the redeemer with updated execution units
    redeemer_cbor: String,
    /// Execution units consumed while evaluating the redeemer
    ex_units: ExUnits,
    /// Computed execution cost for the redeemer (mem, cpu)
    cost: Budget,
    /// Remaining budget after evaluating the redeemer
    remaining_budget: Budget,
    /// Budget that was available when evaluating the redeemer
    initial_budget: Budget,
    /// Any log messages emitted while evaluating the redeemer
    logs: Vec<String>,
    /// Labels emitted while evaluating the redeemer
    labels: Vec<String>,
    /// Optional debug cost information
    debug_cost: Option<Vec<i64>>,
    /// Result returned by the Plutus script
    result: EvalOutcome,
}

#[derive(Debug, Serialize)]
struct ExUnits {
    mem: u64,
    steps: u64,
}

#[derive(Debug, Serialize)]
struct Budget {
    mem: i64,
    cpu: i64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
enum EvalOutcome {
    Success { term: String },
    Failure { error: String },
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("invalid hex in field `{field}`: {source}")]
    InvalidHex {
        field: String,
        #[source]
        source: hex::FromHexError,
    },
    #[error("invalid additional utxo at index {index}: {message}")]
    InvalidAdditionalUtxo { index: usize, message: String },
    #[error("transaction evaluation failed: {0}")]
    Evaluation(#[from] tx::error::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match self {
            AppError::InvalidHex { .. } => StatusCode::BAD_REQUEST,
            AppError::InvalidAdditionalUtxo { .. } => StatusCode::BAD_REQUEST,
            AppError::Evaluation(_) => StatusCode::BAD_REQUEST,
        };

        let body = Json(ErrorResponse {
            error: self.to_string(),
        });

        (status, body).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info"))?;
    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let addr: SocketAddr = env::var("SERVER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string())
        .parse()?;

    let blockfrost_project_id = env::var("BLOCKFROST_API_KEY")
        .context("BLOCKFROST_API_KEY env var must be set")?;
    let blockfrost_api = BlockfrostAPI::new(blockfrost_project_id.as_str(), Default::default());

    let protocol_parameters = match blockfrost_api.epochs_latest_parameters().await {
        Ok(params) => {
            info!("successfully fetched protocol parameters from Blockfrost");
            params
        }
        Err(e) => {
            tracing::error!("CRITICAL: failed to fetch protocol parameters from Blockfrost: {e}");
            return Err(format!(
                "server startup aborted: cannot fetch protocol parameters from Blockfrost: {e}"
            )
            .into());
        }
    };

    let blockfrost_api = Arc::new(blockfrost_api);

    let max_tx_ex_steps = protocol_parameters
        .max_tx_ex_steps
        .as_ref()
        .ok_or("server startup aborted: max_tx_ex_steps not defined in protocol parameters")?
        .parse::<u64>()
        .map_err(|e| format!("server startup aborted: failed to parse max_tx_ex_steps: {e}"))?;

    let max_tx_ex_mem = protocol_parameters
        .max_tx_ex_mem
        .as_ref()
        .ok_or("server startup aborted: max_tx_ex_mem not defined in protocol parameters")?
        .parse::<u64>()
        .map_err(|e| format!("server startup aborted: failed to parse max_tx_ex_mem: {e}"))?;

    info!("using protocol parameters: max_tx_ex_steps={max_tx_ex_steps}, max_tx_ex_mem={max_tx_ex_mem}");

    let initial_budget = (max_tx_ex_steps, max_tx_ex_mem);
    let slot_config = (
        env_u64("SLOT_CONFIG_ZERO_TIME", 1_596_059_091_000),
        env_u64("SLOT_CONFIG_ZERO_SLOT", 4_492_800),
        env_u32("SLOT_CONFIG_SLOT_LENGTH", 1_000),
    );
    let run_phase_one = env_bool("RUN_PHASE_ONE", false);
    let cost_models_hex = cost_models_hex(&protocol_parameters)
        .context("failed to encode cost models")?
        .ok_or("server startup aborted: no cost models configured in protocol parameters")?;

    let cost_models_bytes = hex::decode(&cost_models_hex)
        .map_err(|e| format!("server startup aborted: failed to decode cost models hex: {e}"))?;

    info!("successfully loaded and encoded cost models");

    let state = AppState {
        initial_budget,
        slot_config,
        run_phase_one,
        cost_models: Arc::new(cost_models_bytes),
        blockfrost_api,
    };

    let app = Router::new()
        .route("/evaluate", post(evaluate_transaction))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn evaluate_transaction(
    State(state): State<AppState>,
    State(_blockfrost_api): State<Arc<BlockfrostAPI>>,
    Json(payload): Json<EvaluationRequest>,
) -> Result<Json<EvaluationResponse>, AppError> {

    let tx_bytes = decode_hex("cbor", &payload.cbor)?;

    let utxos = additional_utxos_to_cbor(&payload.additional_utxo_set)?;

    let initial_budget = state.initial_budget;
    let slot_config = state.slot_config;
    let cost_models = Some(state.cost_models.as_ref().as_slice());

    let results = tx::eval_phase_two_raw(
        &tx_bytes,
        &utxos,
        cost_models,
        initial_budget,
        slot_config,
        state.run_phase_one,
        |_| (),
    )?;

    let mut redeemers = Vec::with_capacity(results.len());

    for (redeemer_bytes, eval) in results {
        let logs = eval.logs();
        let labels = eval.labels();
        let cost_budget = eval.cost();
        let debug_cost = eval.debug_cost();
        let result_term = eval.result();

        let uplc::machine::eval_result::EvalResult {
            remaining_budget,
            initial_budget,
            ..
        } = eval;

        let ex_units = ExUnits {
            mem: positive_u64(cost_budget.mem),
            steps: positive_u64(cost_budget.cpu),
        };

        let response = RedeemerEvaluation {
            redeemer_cbor: hex::encode(redeemer_bytes),
            ex_units,
            cost: Budget {
                mem: cost_budget.mem,
                cpu: cost_budget.cpu,
            },
            remaining_budget: Budget {
                mem: remaining_budget.mem,
                cpu: remaining_budget.cpu,
            },
            initial_budget: Budget {
                mem: initial_budget.mem,
                cpu: initial_budget.cpu,
            },
            logs,
            labels,
            debug_cost,
            result: match result_term {
                Ok(term) => EvalOutcome::Success {
                    term: format!("{term:?}"),
                },
                Err(err) => EvalOutcome::Failure {
                    error: err.to_string(),
                },
            },
        };

        redeemers.push(response);
    }

    Ok(Json(EvaluationResponse { redeemers }))
}

fn additional_utxos_to_cbor(
    utxos: &[(TxIn, TxOut)],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>, AppError> {
    utxos
        .iter()
        .enumerate()
        .map(|(index, (tx_in, tx_out))| {
            convert_utxo(tx_in, tx_out).map_err(|error| AppError::InvalidAdditionalUtxo {
                index,
                message: error.to_string(),
            })
        })
        .collect()
}

fn convert_utxo(tx_in: &TxIn, tx_out: &TxOut) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let tx_hash = TransactionHash::from_hex(clean_hex(&tx_in.tx_id))
        .map_err(|e| anyhow!("invalid transaction hash `{}`: {e}", tx_in.tx_id))?;
    let input_cbor = TransactionInput::new(tx_hash, tx_in.index as u64).to_cbor_bytes();
    let output_cbor = tx_output_to_cbor(tx_out)?;
    Ok((input_cbor, output_cbor))
}

fn tx_output_to_cbor(tx_out: &TxOut) -> anyhow::Result<Vec<u8>> {
    let address = Address::from_bech32(&tx_out.address)
        .map_err(|e| anyhow!("invalid bech32 address `{}`: {e}", tx_out.address))?;
    let mut builder = TransactionOutputBuilder::new().with_address(address);

    if let Some(hash_hex) = &tx_out.datum_hash {
        let hash = DatumHash::from_hex(clean_hex(hash_hex))
            .map_err(|e| anyhow!("invalid datum hash `{hash_hex}`: {e}"))?;
        builder = builder.with_data(DatumOption::new_hash(hash));
    } else if let Some(datum_field) = &tx_out.datum {
        let datum = datum_from_field(datum_field)?;
        builder = builder.with_data(DatumOption::new_datum(datum));
    }

    if let Some(script) = &tx_out.script {
        let script_ref = script_ref_from_json(script)?;
        builder = builder.with_reference_script(script_ref);
    }

    let value = value_from_json(&tx_out.value)?;
    let output = builder
        .next()
        .context("address missing when building transaction output")?
        .with_value(value)
        .build()
        .context("failed to build transaction output")?
        .output;

    Ok(output.to_cbor_bytes())
}

fn value_from_json(value: &ValueJson) -> anyhow::Result<CmlValue> {
    let mut multiasset = MultiAsset::new();
    if let Some(assets) = &value.assets {
        for (asset_id, amount) in assets {
            let (policy_hex, name_hex) = asset_id.split_once('.').ok_or_else(|| {
                anyhow!("asset id `{asset_id}` must be in the form policyId.assetName")
            })?;

            let policy = PolicyId::from_hex(clean_hex(policy_hex))
                .map_err(|e| anyhow!("invalid policy id `{policy_hex}`: {e}"))?;
            let name_bytes = hex::decode(clean_hex(name_hex))
                .with_context(|| format!("asset name `{name_hex}` must be hex"))?;
            let asset_name = AssetName::new(name_bytes)
                .map_err(|e| anyhow!("asset name `{name_hex}` is invalid: {e}"))?;

            multiasset.set(policy, asset_name, *amount);
        }
    }

    Ok(CmlValue::new(value.coins, multiasset))
}

fn script_ref_from_json(script: &ScriptJson) -> anyhow::Result<Script> {
    match script {
        ScriptJson::PlutusV1 { hex } => {
            let bytes = hex::decode(clean_hex(hex))
                .with_context(|| "Plutus V1 script must be hex encoded".to_string())?;
            let script = PlutusV1Script::from_raw_bytes(&bytes)
                .map_err(|e| anyhow!("invalid Plutus V1 script bytes: {e}"))?;
            Ok(Script::new_plutus_v1(script))
        }
        ScriptJson::PlutusV2 { hex } => {
            let bytes = hex::decode(clean_hex(hex))
                .with_context(|| "Plutus V2 script must be hex encoded".to_string())?;
            let script = PlutusV2Script::from_raw_bytes(&bytes)
                .map_err(|e| anyhow!("invalid Plutus V2 script bytes: {e}"))?;
            Ok(Script::new_plutus_v2(script))
        }
        ScriptJson::PlutusV3 { hex } => {
            let bytes = hex::decode(clean_hex(hex))
                .with_context(|| "Plutus V3 script must be hex encoded".to_string())?;
            let script = PlutusV3Script::from_raw_bytes(&bytes)
                .map_err(|e| anyhow!("invalid Plutus V3 script bytes: {e}"))?;
            Ok(Script::new_plutus_v3(script))
        }
    }
}

fn datum_from_field(field: &DatumField) -> anyhow::Result<PlutusData> {
    match field {
        DatumField::Hex(hex_value) => {
            let bytes = hex::decode(clean_hex(hex_value))
                .with_context(|| "inline datum must be hex encoded CBOR".to_string())?;
            PlutusData::from_cbor_bytes(&bytes)
                .map_err(|e| anyhow!("invalid CBOR inline datum: {e}"))
        }
        DatumField::Object(value) => serde_json::from_value::<PlutusData>(value.clone())
            .context("invalid JSON representation for inline datum"),
    }
}

fn clean_hex(value: &str) -> &str {
    let trimmed = value.trim();
    trimmed.strip_prefix("0x").unwrap_or(trimmed)
}

fn decode_hex(field: impl Into<String>, value: &str) -> Result<Vec<u8>, AppError> {
    let cleaned = value.trim().trim_start_matches("0x");
    hex::decode(cleaned).map_err(|source| AppError::InvalidHex {
        field: field.into(),
        source,
    })
}

fn positive_u64(value: i64) -> u64 {
    if value < 0 {
        0
    } else {
        value as u64
    }
}

fn env_u64(var: &str, default: u64) -> u64 {
    env::var(var)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u32(var: &str, default: u32) -> u32 {
    env::var(var)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(var: &str, default: bool) -> bool {
    env::var(var)
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn cost_models_hex(params: &EpochParamContent) -> anyhow::Result<Option<String>> {
    let Some(raw_models) = &params.cost_models else {
        return Ok(None); // nothing to encode
    };

    let mut models = CostModels::default();

    let mut languages = raw_models.iter().collect::<Vec<_>>();
    languages.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));

    for (lang_name, costs_json) in languages {
        let language = match lang_name.as_str() {
            "PlutusV1" => Language::PlutusV1,
            "PlutusV2" => Language::PlutusV2,
            "PlutusV3" => Language::PlutusV3,
            other => return Err(anyhow!("unknown cost model language: {other}")),
        };

        let obj = costs_json
            .as_object()
            .context("cost model must be a JSON object")?;
        let mut entries = obj.iter().collect::<Vec<_>>();
        entries.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));

        let mut costs = Vec::with_capacity(entries.len());
        for (_, value) in entries {
            costs.push(
                value
                    .as_i64()
                    .context("cost model entries must be integers")?,
            );
        }

        models.inner.insert(language.into(), costs);
    }

    let bytes = encode_language_views(&models).context("unable to encode language views")?;
    Ok(Some(hex::encode(bytes)))
}

fn encode_language_views(models: &CostModels) -> anyhow::Result<Vec<u8>> {
    use cbor_event::{self, se::Serializer};

    // Sort by language id to enforce canonical ordering.
    let mut entries = models.inner.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(language_id, _)| *language_id);

    let mut serializer = Serializer::new_vec();
    serializer.write_map(cbor_event::Len::Len(entries.len() as u64))?;

    for (language_id, costs) in entries {
        match *language_id {
            0 => {
                // PlutusV1 uses a special encoding that nests an indefinite length list inside a bytestring.
                serializer.write_bytes(&[0])?;

                let mut cost_serializer = Serializer::new_vec();
                cost_serializer.write_array(cbor_event::Len::Indefinite)?;
                for cost in costs {
                    if *cost >= 0 {
                        cost_serializer.write_unsigned_integer(cost.unsigned_abs())?;
                    } else {
                        cost_serializer.write_negative_integer(*cost)?;
                    }
                }
                cost_serializer.write_special(cbor_event::Special::Break)?;

                serializer.write_bytes(cost_serializer.finalize())?;
            }
            language => {
                serializer.write_unsigned_integer(language)?;
                serializer.write_array(cbor_event::Len::Len(costs.len() as u64))?;
                for cost in costs {
                    if *cost >= 0 {
                        serializer.write_unsigned_integer(cost.unsigned_abs())?;
                    } else {
                        serializer.write_negative_integer(*cost)?;
                    }
                }
            }
        }
    }

    Ok(serializer.finalize())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut term = signal(SignalKind::terminate()).expect("failed to install signal handler");

        tokio::select! {
            _ = signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }

    #[cfg(not(unix))]
    {
        signal::ctrl_c()
            .await
            .expect("failed to install ctrl+c handler");
    }
}
