set -euo pipefail

REPO_DIR="/home/antn/RNALoc-LM"
ENV_NAME="RNALoc-LM"
LOG_DIR="${REPO_DIR}/logs"
RUN_LOG="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"


CONDA_BASE="$(conda info --base)"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

log() { echo "[$(date '+%F %T')] $*" | tee -a "${RUN_LOG}"; }

run_one() {
    local rna="$1"
    local script="${rna}.py"
    local out="${LOG_DIR}/run_${rna}_$(date +%Y%m%d_%H%M%S).log"
    local start end elapsed

    log "=== START ${rna} (script: ${script}) ==="
    log "stdout/stderr -> ${out}"
    start=$(date +%s)

    if ! python -u "${script}" >"${out}" 2>&1; then
        log "!!! FAIL ${rna} — xem ${out}"
        tail -n 40 "${out}" | tee -a "${RUN_LOG}"
        exit 1
    fi

    end=$(date +%s)
    elapsed=$((end - start))
    log "=== DONE  ${rna} sau $((elapsed/60))m$((elapsed%60))s ==="
}

log "Repo:   ${REPO_DIR}"
log "Env:    ${ENV_NAME}  (python: $(python --version 2>&1))"
log "GPU:    $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
log "Run log: ${RUN_LOG}"

global_start=$(date +%s)
run_one miRNA
run_one lncRNA
run_one circRNA
global_end=$(date +%s)
total=$((global_end - global_start))

log "ALL DONE in $((total/3600))h$(((total%3600)/60))m$((total%60))s"
log "Checkpoints:"
find models -name "*.pkl" | sort | tee -a "${RUN_LOG}"
