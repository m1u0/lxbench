#!/usr/bin/env bash

usage() {
    printf 'usage: run.sh --endpoint URL --requests PATH --output PATH\n' >&2
}

endpoint=
requests_path=
output_path=

while (($# > 0)); do
    case $1 in
        --endpoint|--requests|--output)
            if (($# < 2)); then
                usage
                exit 2
            fi
            case $1 in
                --endpoint) endpoint=$2 ;;
                --requests) requests_path=$2 ;;
                --output) output_path=$2 ;;
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
    request_lines[${#request_lines[@]}]=$line
done < "$requests_path"

ids=()
while IFS= read -r line || [[ -n $line ]]; do
    ids[${#ids[@]}]=$line
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
    prepared_ids[${#prepared_ids[@]}]=$prepared_id
done

completed_ids=()
needs_separator=0
if [[ -e $output_path ]]; then
    if [[ ! -r $output_path ]]; then
        printf 'error: cannot read raw run %s\n' "$output_path" >&2
        exit 1
    fi
    output_lines=()
    while IFS= read -r line || [[ -n $line ]]; do
        output_lines[${#output_lines[@]}]=$line
    done < "$output_path"
    for line in "${output_lines[@]}"; do
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
        completed_ids[${#completed_ids[@]}]=$completed_id
    done

    raw_output=
    IFS= read -r -d '' raw_output < "$output_path" || true
    if [[ -n $raw_output && $raw_output != *$'\n' ]]; then
        needs_separator=1
    fi
fi

if [[ $output_path == */* ]]; then
    output_parent=${output_path%/*}
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
    local attempt combined curl_status http_status retryable wait_seconds

    for attempt in 0 1 2 3; do
        combined=
        if combined=$(curl --silent \
            --request POST \
            --header 'Content-Type: application/json' \
            --data-binary "$request_body" \
            --write-out $'\n%{http_code}' \
            "$endpoint"); then
            http_status=${combined##*$'\n'}
            response_body=${combined%$'\n'*}
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
        case $attempt in
            0) wait_seconds=1 ;;
            1) wait_seconds=2 ;;
            2) wait_seconds=4 ;;
        esac
        sleep "$wait_seconds"
    done

    return 1
}

failed=0
for index in "${!ids[@]}"; do
    prepared_id=${ids[$index]}
    if contains_id "$prepared_id" "${completed_ids[@]}"; then
        continue
    fi
    if ! post_with_retries "${request_lines[$index]}"; then
        printf '%s: %s\n' "$prepared_id" "$failure_reason" >&2
        failed=1
        continue
    fi
    if ((needs_separator)); then
        printf '\n' >> "$output_path"
        needs_separator=0
    fi
    if ! printf '{"id":"%s","response":%s}\n' \
        "$prepared_id" "$response_body" >> "$output_path"; then
        printf 'error: cannot append to raw run %s\n' "$output_path" >&2
        exit 1
    fi
done

exit "$failed"
