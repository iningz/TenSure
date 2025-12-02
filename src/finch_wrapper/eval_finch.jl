import LazyJSON as JSON
using Finch

"""
    compile_format(formats)

Converts a list of format strings (e.g., ["dense", "compressed"]) into a
Finch Level constructor expression (e.g., :(Dense(SparseList(Element(0.0))))).
"""
function compile_format(formats)
    # Fill-value is Element(0.0)
    expr = :(Element(0.0))

    # Iterate in reverse order (inner to outer) to wrap levels
    for fmt in reverse(collect(formats))
        s_fmt = String(fmt)
        if s_fmt == "dense"
            expr = :(Dense($expr))
        elseif s_fmt == "compressed"
            expr = :(SparseList($expr))
        else
            # Default or fallback
            expr = :(Dense($expr))
        end
    end
    return expr
end

"""
emit_einsum_block(name, spec)
Generates a Julia expression block for a single einsum operation defined in the JSON spec.
"""
function emit_einsum_block(name, spec)
    input_defs = []
    input_terms = []

    # Parse einsum string: "ik,kj->ij"
    einsum_str = String(spec["einsum"])
    inputs_indices_str, out_indices_str = split(einsum_str, "->")

    # Parse indices into Symbols: "ik" -> [:i, :k]
    parse_indices(str) = [Symbol(char) for char in str]

    input_indices_list = map(parse_indices, split(inputs_indices_str, ","))
    out_indices = parse_indices(out_indices_str)

    inputs = spec["inputs"]

    for (i, input) in enumerate(inputs)
        var_name = Symbol(String(input["name"]))
        file_path = String(input["file"])
        formats = input["format"]

        fmt_expr = compile_format(formats)

        # Generate: B = Tensor(Dense(SparseList(...)), fread("path"))
        push!(input_defs, :($var_name = Tensor($fmt_expr, fread($file_path))))

        # Prepare access term: B[i, k]
        indices = input_indices_list[i]
        push!(input_terms, :($var_name[$(indices...)]))
    end

    # 2. Prepare Output
    out_spec = spec["output"]
    out_name = Symbol(String(out_spec["name"]))
    out_file = String(out_spec["file"])
    out_fmt = compile_format(out_spec["format"])

    # Initialize output tensor
    out_def = :($out_name = Tensor($out_fmt))

    # Combine input terms: B[i,k] * C[k,j]
    rhs_expr = foldl((a, b) -> :($a * $b), input_terms)

    # Construct LHS access: y[i,j]
    lhs_expr = :($out_name[$(out_indices...)])

    # @einsum y[i,j] += B[i,k] * C[k,j]
    kernel_expr = :(@einsum $lhs_expr += $rhs_expr)

    write_expr = :(fwrite($out_file, $out_name))

    return quote
        $(input_defs...)
        $out_def
        $kernel_expr
        $write_expr
    end
end

function run_eval()
    dump = "--dump" in ARGS
    args = filter(x -> x != "--dump", ARGS)

    if length(args) < 1
        error("Usage: julia eval_finch.jl <json_spec_path> [--dump]")
    end
    spec_path = args[1]
    if !isfile(spec_path)
        error("Could not find $spec_path")
    end

    spec = JSON.value(read(spec_path, String))

    program = quote
        using Finch
        using TensorMarket
    end

    for key in keys(spec)
        key_str = String(key)
        if key_str == "\$schema"
            continue
        end

        kernel_spec = spec[key_str]
        block = emit_einsum_block(key_str, kernel_spec)
        push!(program.args, block)
    end

    if dump
        Base.remove_linenums!(program)
        write(splitext(basename(spec_path))[1] * ".jl", join(program.args, "\n"))
    end

    eval(program)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_eval()
end
