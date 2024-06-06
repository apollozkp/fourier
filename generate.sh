#!/bin/bash

: "${SCALE:=6}"
: "${MACHINES_SCALE:=2}"
: "${UNCOMPRESSED:=false}"
: "${OVERWRITE:=false}"

BIN=./target/release/fourier

# If no parameters are passed, print usage
if [ "$#" -eq 0 ]; then
    echo "Note on logs:"
    echo "export RUST_LOG=info to see logs"
    echo "export RUST_LOG=debug to enable debug mode"
	echo "Usage: $0 [OPTIONS]"
	echo "Options:"
	echo "  -s, --scale <scale>            Scale of the dataset (default: 6)"
	echo "  -m, --machines-scale <scale>   Scale of the machines dataset (default: 2)"
	echo "  -u, --uncompressed             Generate uncompressed files"
	echo "  -o, --overwrite                Overwrite existing files"
	exit 1
fi

# Generate file name
function filename() {
	local name=$1
	local scale=$2
	local suffix=$3
	local ext=$4
	echo "data/${name}_${scale}_${suffix}.${ext}"
}

# Generate files
function generate_files() {
	local scale=$1
	local machines_scale=$2
	local uncompressed=$3
	local overwrite=$4
    
	if [ "$uncompressed" = true ]; then
		local ext="uncompressed"
	else
		local ext="compressed"
	fi

	local setup_file_name=$(filename setup $scale $machines_scale $ext)
	local precompute_file_name=$(filename precompute $scale $machines_scale $ext)

    # Set arguments
	local args="--scale $scale --machines-scale $machines_scale"
    args="$args --setup-path $setup_file_name --precompute-path $precompute_file_name"
    args="$args --generate-setup --generate-precompute"

	if [ "$uncompressed" = true ]; then
		args="$args --uncompressed"
	fi
	if [ "$overwrite" = true ]; then
		args="$args --overwrite"
	fi

    # run the binary
    $BIN setup $args
}

# Parse parameters
while [ "$#" -gt 0 ]; do
    case "$1" in
        -s | --scale)
            SCALE=$2
            shift
            ;;
        -m | --machines-scale)
            MACHINES_SCALE=$2
            shift
            ;;
        -u | --uncompressed)
            UNCOMPRESSED=true
            ;;
        -o | --overwrite)
            OVERWRITE=true
            ;;
        -*)
            for (( i=1; i<${#1}; i++ )); do
                case "${1:$i:1}" in
                    s)
                        SCALE=$2
                        shift
                        ;;
                    m)
                        MACHINES_SCALE=$2
                        shift
                        ;;
                    u)
                        UNCOMPRESSED=true
                        ;;
                    o)
                        OVERWRITE=true
                        ;;
                    *)
                        echo "Unknown option: ${1:$i:1}"
                        exit 1
                        ;;
                esac
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Generate files
generate_files $SCALE $MACHINES_SCALE $UNCOMPRESSED $OVERWRITE
