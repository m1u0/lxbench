#!/bin/bash

usage() {
    printf 'usage: run.sh --endpoint URL --requests PATH --output PATH [--concurrency N]\n' >&2
}

endpoint=
requests_path=
output_path=
concurrency=1
retry_waits=(1 2 4)

while (($# > 0)); do
    case $1 in
        --endpoint|--requests|--output|--concurrency)
            if (($# < 2)); then
                usage
                exit 2
            fi
            case $1 in
                --endpoint) endpoint=$2 ;;
                --requests) requests_path=$2 ;;
                --output) output_path=$2 ;;
                --concurrency) concurrency=$2 ;;
            esac
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z $endpoint || -z $requests_path || -z $output_path ]]; then
    usage
    exit 2
fi
if [[ ! $concurrency =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: concurrency must be a positive integer\n' >&2
    exit 2
fi

if [[ $requests_path == */* ]]; then
    ids_path=${requests_path%/*}/ids.txt
else
    ids_path=ids.txt
fi

if [[ ! -r $requests_path ]]; then
    printf 'error: cannot read requests file %s\n' "$requests_path" >&2
    exit 1
fi
if [[ ! -r $ids_path ]]; then
    printf 'error: cannot read IDs file %s\n' "$ids_path" >&2
    exit 1
fi

request_lines=()
while IFS= read -r line || [[ -n $line ]]; do
    request_lines+=("$line")
done < "$requests_path"

ids=()
while IFS= read -r line || [[ -n $line ]]; do
    ids+=("$line")
done < "$ids_path"

if ((${#request_lines[@]} != ${#ids[@]})); then
    printf 'error: request and ID counts do not match\n' >&2
    exit 1
fi

contains_id() {
    local wanted=$1
    shift
    local candidate
    for candidate in "$@"; do
        [[ $candidate == "$wanted" ]] && return 0
    done
    return 1
}

prepared_ids=()
for index in "${!ids[@]}"; do
    prepared_id=${ids[$index]}
    if [[ -z ${request_lines[$index]} ]]; then
        printf 'error: requests file contains a blank record\n' >&2
        exit 1
    fi
    if [[ ! $prepared_id =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]]; then
        printf 'error: prepared IDs must be nonblank safe IDs\n' >&2
        exit 1
    fi
    if contains_id "$prepared_id" "${prepared_ids[@]}"; then
        printf 'error: prepared IDs must be unique\n' >&2
        exit 1
    fi
    prepared_ids+=("$prepared_id")
done

completed_ids=()
needs_separator=0
if [[ -e $output_path ]]; then
    if [[ ! -r $output_path ]]; then
        printf 'error: cannot read raw run %s\n' "$output_path" >&2
        exit 1
    fi
    while IFS= read -r line || [[ -n $line ]]; do
        if [[ ! $line =~ ^\{[[:space:]]*\"id\"[[:space:]]*:[[:space:]]*\"([A-Za-z0-9][A-Za-z0-9._:-]*)\"[[:space:]]*,[[:space:]]*\"response\"[[:space:]]*:[[:space:]]*\{.*\}[[:space:]]*\}$ ]]; then
            printf 'error: raw run contains a malformed envelope\n' >&2
            exit 1
        fi
        completed_id=${BASH_REMATCH[1]}
        if ! contains_id "$completed_id" "${prepared_ids[@]}"; then
            printf 'error: raw run contains unknown ID %s\n' "$completed_id" >&2
            exit 1
        fi
        if contains_id "$completed_id" "${completed_ids[@]}"; then
            printf 'error: raw run contains duplicate ID %s\n' "$completed_id" >&2
            exit 1
        fi
        completed_ids+=("$completed_id")
    done < "$output_path"

    raw_output=
    IFS= read -r -d '' raw_output < "$output_path" || true
    if [[ -n $raw_output && $raw_output != *$'\n' ]]; then
        needs_separator=1
    fi
fi

if [[ $output_path == */* ]]; then
    output_parent=${output_path%/*}
    [[ -n $output_parent ]] || output_parent=/
else
    output_parent=.
fi
if ! mkdir -p "$output_parent"; then
    printf 'error: cannot create output parent %s\n' "$output_parent" >&2
    exit 1
fi
if ! : >> "$output_path"; then
    printf 'error: cannot append to raw run %s\n' "$output_path" >&2
    exit 1
fi

post_with_retries() {
    local request_body=$1
    local attempt response_and_status curl_status http_status retryable

    for attempt in 0 1 2 3; do
        if [[ -e $worker_dir/interrupted ]]; then
            failure_reason=interrupted
            return 1
        fi
        if response_and_status=$(printf '%s' "$request_body" | curl --silent \
            --request POST \
            --header 'Content-Type: application/json' \
            --data-binary @- \
            --write-out $'\n%{http_code}' \
            "$endpoint"); then
            http_status=${response_and_status##*$'\n'}
            response_body=${response_and_status%$'\n'*}
            if [[ $http_status =~ ^2[0-9][0-9]$ ]]; then
                return 0
            fi
            failure_reason="HTTP $http_status"
            if [[ $http_status == 408 || $http_status == 429 || $http_status =~ ^5[0-9][0-9]$ ]]; then
                retryable=1
            else
                retryable=0
            fi
        else
            curl_status=$?
            failure_reason="network failure (curl exit $curl_status)"
            retryable=1
        fi

        if ((retryable == 0 || attempt == 3)); then
            return 1
        fi
        if [[ -e $worker_dir/interrupted ]]; then
            failure_reason=interrupted
            return 1
        fi
        sleep "${retry_waits[$attempt]}"
    done
}

request_worker() {
    trap '' INT
    local result_path=$worker_dir/$2.result
    if post_with_retries "$1"; then
        printf 'ok\n%s\n' "$response_body" > "$result_path"
    else
        printf 'error\n%s\n' "$failure_reason" > "$result_path"
    fi
    printf '%s\n' "$2" >> "$completion_path"
}

pending_indices=()
for index in "${!ids[@]}"; do
    if ! contains_id "${ids[$index]}" "${completed_ids[@]}"; then
        pending_indices+=("$index")
    fi
done

failed=0
interrupted=0
active=0
next_pending=0
worker_dir=
completion_path=
slot_active=()
slot_indices=()
slot_pids=()
slot_dead=()
job_slots=()

if ((${#pending_indices[@]} > 0)); then
    worker_dir=$output_parent/.lxbench-workers-$$
    if ! mkdir "$worker_dir"; then
        printf 'error: cannot create worker directory %s\n' "$worker_dir" >&2
        exit 1
    fi
    trap 'rm -rf "$worker_dir"' EXIT
    completion_path=$worker_dir/completed
    if ! : > "$completion_path"; then
        printf 'error: cannot create completion queue %s\n' "$completion_path" >&2
        exit 1
    fi
    exec 3< "$completion_path"
fi

launch_worker() {
    local slot=$1 job=$next_pending index=${pending_indices[$next_pending]}
    request_worker "${request_lines[$index]}" "$job" &
    slot_active[$slot]=1
    slot_indices[$slot]=$index
    slot_pids[$slot]=$!
    slot_dead[$slot]=0
    job_slots[$job]=$slot
    ((active += 1))
    ((next_pending += 1))
}

complete_worker() {
    local slot=$1 status=$2 payload=$3
    local index=${slot_indices[$slot]}
    local prepared_id=${ids[$index]}
    wait "${slot_pids[$slot]}" 2>/dev/null || true
    slot_active[$slot]=0
    ((active -= 1))

    if [[ $status == ok ]]; then
        if ((needs_separator)); then
            printf '\n' >> "$output_path"
            needs_separator=0
        fi
        if ! printf '{"id":"%s","response":%s}\n' \
            "$prepared_id" "$payload" >> "$output_path"; then
            printf 'error: cannot append to raw run %s\n' "$output_path" >&2
            exit 1
        fi
    else
        printf '%s: %s\n' "$prepared_id" "$payload" >&2
        failed=1
    fi

    if ((interrupted == 0 && next_pending < ${#pending_indices[@]})); then
        launch_worker "$slot"
    fi
}

interrupt() {
    interrupted=1
    if [[ -n $worker_dir ]]; then
        : > "$worker_dir/interrupted"
    fi
}

trap interrupt INT

slot=0
while ((
    interrupted == 0 &&
    slot < concurrency &&
    next_pending < ${#pending_indices[@]}
)); do
    launch_worker "$slot"
    ((slot += 1))
done

while ((active > 0)); do
    made_progress=0
    job=
    if IFS= read -r -u 3 job; then
        slot=${job_slots[$job]}
        result_path=$worker_dir/$job.result
        status=
        payload=
        if ! { IFS= read -r status && IFS= read -r payload; } < "$result_path"; then
            status=error
            payload='worker exited before returning a result'
        fi
        complete_worker "$slot" "$status" "$payload"
        made_progress=1
    else
        for ((slot = 0; slot < concurrency; slot += 1)); do
            [[ ${slot_active[$slot]:-0} == 1 ]] || continue
            if kill -0 "${slot_pids[$slot]}" 2>/dev/null; then
                slot_dead[$slot]=0
                continue
            fi
            if ((${slot_dead[$slot]:-0})); then
                complete_worker "$slot" error 'worker exited before returning a result'
                made_progress=1
                break
            fi
            slot_dead[$slot]=1
        done
    fi
    if ((made_progress == 0)); then
        sleep 0.05
    fi
done

trap - INT
if [[ -n $worker_dir ]]; then
    exec 3<&-
fi
if ((interrupted)); then
    exit 130
fi
exit "$failed"
