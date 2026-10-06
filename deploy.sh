#!/usr/bin/env bash
# Build with Cloud Build into Artifact Registry and deploy to Cloud Run.
# Needs gcloud authenticated with a project set. The access code is generated
# once into .secrets/access_code.txt (gitignored) and never printed.
set -euo pipefail
cd "$(dirname "$0")"

GCLOUD="${GCLOUD:-$HOME/google-cloud-sdk/bin/gcloud}"
PROJECT="${GCP_PROJECT:-booming-entity-510706-e9}"
REGION="${GCP_REGION:-asia-southeast1}"
SERVICE=streamclearance
REPO=streamclearance
SA="hercules-deployer@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo -dirty)"

CODE_FILE=.secrets/access_code.txt
if [[ ! -s "$CODE_FILE" ]]; then
  mkdir -p .secrets && chmod 700 .secrets
  (umask 077; python3 -c 'import secrets, string; a = string.ascii_letters + string.digits; print("".join(secrets.choice(a) for _ in range(10)), end="")' >"$CODE_FILE")
  echo "generated a new access code in $CODE_FILE"
fi
ACCESS_CODE="$(cat "$CODE_FILE")"

if ! "$GCLOUD" artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  "$GCLOUD" artifacts repositories create "$REPO" --repository-format docker \
    --location "$REGION" --project "$PROJECT" --description "StreamClearance images"
fi

# The image has no .git: record the app sha and law-file dates for report provenance.
PY=python3; [[ -x .venv/bin/python ]] && PY=.venv/bin/python
"$PY" -m app.provenance

# Logs go to our own bucket: the default one needs project Viewer to stream.
"$GCLOUD" builds submit --project "$PROJECT" --region "$REGION" --tag "$IMAGE" \
  --gcs-log-dir "gs://${PROJECT}_cloudbuild/logs" .

"$GCLOUD" run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "$SA" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=TRUE,SC_APP_REGION=${REGION},SC_REPORT_TTL_DAYS=${SC_REPORT_TTL_DAYS:-7},SC_ACCESS_CODE=${ACCESS_CODE}" \
  --memory 2Gi --cpu 2 --no-cpu-throttling --cpu-boost --timeout 900 \
  --min-instances 0 --max-instances 1 --concurrency 20 \
  --allow-unauthenticated --quiet

"$GCLOUD" run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)'
