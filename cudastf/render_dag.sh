#!/usr/bin/env bash

set -euo pipefail

usage()
{
  cat <<'EOF'
Usage:
  render_dag.sh OUTPUT.{png,pdf,svg} [--] [TASK_BENCH_ARGUMENTS...]

Runs one CUDASTF sample, saves the generated DOT file beside OUTPUT, and
renders OUTPUT with Graphviz. CUDASTF prerequisite nodes are hidden by
default; set CUDASTF_DOT_IGNORE_PREREQS=0 to include them.
EOF
}

if (($# == 0)) || [[ $1 == "-h" || $1 == "--help" ]]; then
  usage
  exit $(( $# == 0 ))
fi

output_path=$1
shift
if (($# > 0)) && [[ $1 == "--" ]]; then
  shift
fi

extension=${output_path##*.}
format=${extension,,}
case "$format" in
  png|pdf|svg) ;;
  *)
    echo "unsupported output format: '$extension' (expected png, pdf, or svg)" >&2
    exit 2
    ;;
esac

output_directory=$(dirname "$output_path")
if [[ ! -d $output_directory ]]; then
  echo "output directory does not exist: $output_directory" >&2
  exit 2
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_bench="$script_directory/task_bench"
if [[ ! -x $task_bench ]]; then
  echo "CUDASTF Task Bench is not built: $task_bench" >&2
  echo "Run 'make -C $script_directory' first." >&2
  exit 2
fi

graphviz_dot=$(command -v dot || true)
if [[ -z $graphviz_dot ]]; then
  echo "Graphviz 'dot' was not found in PATH" >&2
  exit 2
fi

dot_path="${output_path%.*}.dot"
CUDASTF_DOT_FILE="$dot_path" \
CUDASTF_DOT_IGNORE_PREREQS="${CUDASTF_DOT_IGNORE_PREREQS:-1}" \
  "$task_bench" "$@" -cuda-warmup 0 -cuda-runs 1

if [[ ! -s $dot_path ]]; then
  echo "CUDASTF did not produce a DOT file: $dot_path" >&2
  exit 1
fi

"$graphviz_dot" -T"$format" "$dot_path" -o "$output_path"

echo "DOT graph: $dot_path"
echo "Rendered graph: $output_path"
