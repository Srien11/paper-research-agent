#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="/srv/paper-research-agent"
SHARED_ROOT="${APP_ROOT}/shared"
SHARED_RUNTIME="${SHARED_ROOT}/runtime"
SHARED_CORPUS="${SHARED_ROOT}/corpus"
SHARED_KNOWLEDGE="${SHARED_ROOT}/knowledge"
CONFIG_ROOT="/etc/paper-research-agent"
SERVICE_NAME="paper-research-agent.service"
NGINX_ZONES_TARGET="/etc/nginx/conf.d/paper-research-agent-zones.conf"
NGINX_LOCATIONS_TARGET="/etc/nginx/snippets/paper-research-agent.conf"
NGINX_INCLUDE="include /etc/nginx/snippets/paper-research-agent.conf;"
ENV_FILE="${CONFIG_ROOT}/paper-research-agent.env"
BUNDLE_PATH="${1:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RELEASE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RELEASE_DIR="${APP_ROOT}/releases/${RELEASE_TAG}"
BACKUP_DIR="${APP_ROOT}/deploy-backups/${RELEASE_TAG}"
LIST_FILE="$(mktemp)"
PREVIOUS_TARGET=""
SWITCHED=0
SERVICE_EXISTED=0
ZONES_EXISTED=0
LOCATIONS_EXISTED=0

cleanup() {
    rm -f -- "${LIST_FILE}"
}

initialize_shared_tree() {
    source_path="$1"
    target_path="$2"
    case "${source_path}" in
        "${APP_ROOT}"/*) ;;
        *) printf 'Unsafe shared data source: %s\n' "${source_path}" >&2; return 1 ;;
    esac
    case "${target_path}" in
        "${SHARED_ROOT}"/*) ;;
        *) printf 'Unsafe shared data target: %s\n' "${target_path}" >&2; return 1 ;;
    esac
    if [[ -e "${target_path}" ]]; then
        if [[ -d "${source_path}" && "$(readlink -f "${source_path}")" != "$(readlink -f "${target_path}")" ]]; then
            cp -a -- "${source_path}/." "${target_path}/"
        fi
        return
    fi
    staging_path="${target_path}.init-${RELEASE_TAG}"
    install -d -m 0700 -o paper-rag -g paper-rag "${staging_path}"
    if [[ -d "${source_path}" ]]; then
        cp -a -- "${source_path}/." "${staging_path}/"
    fi
    mv -- "${staging_path}" "${target_path}"
}

rollback() {
    exit_code=$?
    trap - ERR
    if [[ ${SWITCHED} -eq 1 && -n "${PREVIOUS_TARGET}" && -d "${PREVIOUS_TARGET}" ]]; then
        ln -sfn -- "${PREVIOUS_TARGET}" "${APP_ROOT}/current.rollback"
        mv -Tf -- "${APP_ROOT}/current.rollback" "${APP_ROOT}/current"
    fi
    if [[ -f "${BACKUP_DIR}/${SERVICE_NAME}" ]]; then
        install -m 0644 "${BACKUP_DIR}/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
    elif [[ ${SERVICE_EXISTED} -eq 0 ]]; then
        rm -f -- "/etc/systemd/system/${SERVICE_NAME}"
    fi
    if [[ -f "${BACKUP_DIR}/paper-research-agent-zones.conf" ]]; then
        install -m 0644 "${BACKUP_DIR}/paper-research-agent-zones.conf" "${NGINX_ZONES_TARGET}"
    elif [[ ${ZONES_EXISTED} -eq 0 ]]; then
        rm -f -- "${NGINX_ZONES_TARGET}"
    fi
    if [[ -f "${BACKUP_DIR}/paper-research-agent-locations.conf" ]]; then
        install -m 0644 "${BACKUP_DIR}/paper-research-agent-locations.conf" "${NGINX_LOCATIONS_TARGET}"
    elif [[ ${LOCATIONS_EXISTED} -eq 0 ]]; then
        rm -f -- "${NGINX_LOCATIONS_TARGET}"
    fi
    systemctl daemon-reload || true
    systemctl restart "${SERVICE_NAME}" || true
    nginx -t && systemctl reload nginx || true
    cleanup
    printf 'Deployment failed; previous release restored when available.\n' >&2
    exit "${exit_code}"
}

trap cleanup EXIT
trap rollback ERR

if [[ ${EUID} -ne 0 ]]; then
    printf 'Run this deployment script as root.\n' >&2
    exit 2
fi
if [[ -z "${BUNDLE_PATH}" || ! -f "${BUNDLE_PATH}" ]]; then
    printf 'Usage: %s /absolute/path/private-bundle.tar.gz\n' "$0" >&2
    exit 2
fi
if [[ ! -f "${ENV_FILE}" ]]; then
    printf 'Required environment file is missing: %s\n' "${ENV_FILE}" >&2
    printf 'Create it out of band with mode 600; it must never be included in the bundle.\n' >&2
    exit 2
fi
if [[ "$(stat -c '%a' "${ENV_FILE}")" != "600" ]]; then
    printf 'Environment file must have mode 600.\n' >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'Python interpreter not found: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    printf 'Python 3.11 or newer is required: %s\n' "${PYTHON_BIN}" >&2
    exit 2
fi

tar -tzf "${BUNDLE_PATH}" > "${LIST_FILE}"
if grep -Eiq '(^|/)\.env($|[./])|\.pdf$|\.(pem|key)$|data/runtime/.+\.(sqlite|sqlite3|db)$' "${LIST_FILE}"; then
    printf 'Bundle rejected: secrets, PDFs, keys, or runtime databases are present.\n' >&2
    exit 3
fi
if grep -Eq '(^|/)\.\.?(/|$)' "${LIST_FILE}"; then
    printf 'Bundle rejected: unsafe archive path.\n' >&2
    exit 3
fi

if ! id paper-rag >/dev/null 2>&1; then
    useradd --system --home-dir "${APP_ROOT}" --shell /usr/sbin/nologin paper-rag
fi
install -d -m 0750 -o paper-rag -g paper-rag "${APP_ROOT}/releases" "${APP_ROOT}/deploy-backups" "${SHARED_ROOT}"
install -d -m 0750 -o paper-rag -g paper-rag "${APP_ROOT}/model-cache"
install -d -m 0750 -o root -g paper-rag "${CONFIG_ROOT}"
install -d -m 0750 -o paper-rag -g paper-rag "${RELEASE_DIR}"
install -d -m 0700 -o root -g root "${BACKUP_DIR}"
install -d -m 0755 -o root -g root "/etc/nginx/snippets"

if [[ -L "${APP_ROOT}/current" ]]; then
    PREVIOUS_TARGET="$(readlink -f "${APP_ROOT}/current")"
fi
if [[ -f "/etc/systemd/system/${SERVICE_NAME}" ]]; then
    SERVICE_EXISTED=1
    cp -a -- "/etc/systemd/system/${SERVICE_NAME}" "${BACKUP_DIR}/${SERVICE_NAME}"
fi
if [[ -f "${NGINX_ZONES_TARGET}" ]]; then
    ZONES_EXISTED=1
    cp -a -- "${NGINX_ZONES_TARGET}" "${BACKUP_DIR}/paper-research-agent-zones.conf"
fi
if [[ -f "${NGINX_LOCATIONS_TARGET}" ]]; then
    LOCATIONS_EXISTED=1
    cp -a -- "${NGINX_LOCATIONS_TARGET}" "${BACKUP_DIR}/paper-research-agent-locations.conf"
fi

tar -xzf "${BUNDLE_PATH}" --strip-components=1 -C "${RELEASE_DIR}"
test -f "${RELEASE_DIR}/pyproject.toml"
test -f "${RELEASE_DIR}/scripts/serve_web.py"
test -f "${RELEASE_DIR}/data/processed/chunks/chunks.jsonl"
test -f "${RELEASE_DIR}/data/indexes/retrieval-v1/manifest.json"

runtime_source="${RELEASE_DIR}/data/runtime"
corpus_source="${RELEASE_DIR}/corpus"
knowledge_source=""
if [[ -n "${PREVIOUS_TARGET}" ]]; then
    runtime_source="${PREVIOUS_TARGET}/data/runtime"
    corpus_source="${PREVIOUS_TARGET}/corpus"
    if [[ -f "${PREVIOUS_TARGET}/data/processed/current_knowledge_base.json" ]]; then
        knowledge_source="${PREVIOUS_TARGET}/data/processed"
    fi
fi
if [[ ! -e "${SHARED_RUNTIME}" || ! -e "${SHARED_CORPUS}" || ! -e "${SHARED_KNOWLEDGE}" \
    || "$(readlink -f "${runtime_source}")" != "$(readlink -f "${SHARED_RUNTIME}")" \
    || "$(readlink -f "${corpus_source}")" != "$(readlink -f "${SHARED_CORPUS}")" ]]; then
    systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
fi
initialize_shared_tree "${runtime_source}" "${SHARED_RUNTIME}"
initialize_shared_tree "${corpus_source}" "${SHARED_CORPUS}"
initialize_shared_tree "${knowledge_source:-${RELEASE_DIR}/data/runtime}" "${SHARED_KNOWLEDGE}"

rmdir -- "${RELEASE_DIR}/data/runtime"
ln -s -- "${SHARED_RUNTIME}" "${RELEASE_DIR}/data/runtime"
mv -- "${RELEASE_DIR}/corpus" "${RELEASE_DIR}/corpus.bundle"
ln -s -- "${SHARED_CORPUS}" "${RELEASE_DIR}/corpus"

"${PYTHON_BIN}" -m venv "${RELEASE_DIR}/.venv"
"${RELEASE_DIR}/.venv/bin/python" -m pip install --disable-pip-version-check --no-input "${RELEASE_DIR}[retrieval,web,agent]"
chown -R paper-rag:paper-rag "${RELEASE_DIR}"
chown -R paper-rag:paper-rag "${SHARED_ROOT}"
chmod -R o-rwx "${RELEASE_DIR}"

install -m 0644 "${RELEASE_DIR}/deploy/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
if ! grep -RqsF -- "${NGINX_INCLUDE}" /etc/nginx/sites-enabled /etc/nginx/conf.d; then
    printf 'Existing site config must include: %s\n' "${NGINX_INCLUDE}" >&2
    printf 'Refusing to switch releases or guess-edit the active server block.\n' >&2
    false
fi
install -m 0644 "${RELEASE_DIR}/deploy/nginx-paper-research-zones.conf" "${NGINX_ZONES_TARGET}"
install -m 0644 "${RELEASE_DIR}/deploy/nginx-paper-research-locations.conf" "${NGINX_LOCATIONS_TARGET}"
nginx -t

ln -sfn -- "${RELEASE_DIR}" "${APP_ROOT}/current.next"
mv -Tf -- "${APP_ROOT}/current.next" "${APP_ROOT}/current"
SWITCHED=1
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

for attempt in {1..90}; do
    if curl --fail --silent --show-error --max-time 3 "http://127.0.0.1:8092/paper-research/readyz" >/dev/null; then
        break
    fi
    if [[ ${attempt} -eq 90 ]]; then
        printf 'Readiness check timed out.\n' >&2
        false
    fi
    sleep 2
done

nginx -t
systemctl reload nginx
printf 'Deployed release %s. Previous release: %s\n' "${RELEASE_TAG}" "${PREVIOUS_TARGET:-none}"
