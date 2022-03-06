def stylize_model_arch_for_figures(arch: str) -> str:
    toks = arch.split("+")
    for i in range(1, len(toks)):
        toks[i] = toks[i].upper()
    name = "+".join(toks)
    name = name.replace("roberta", "RoBERTa")
    name = name.replace("logreg", "LogReg")
    name = name.replace("+KB", "+DsBias")
    name = name.replace("+SN", "+DsNorm")
    return name
