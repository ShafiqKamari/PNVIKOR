import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NeutroForge: PNVIKOR Decision Suite")
st.title("NeutroForge")
st.subheader("A Pythagorean Neutrosophic VIKOR Decision Suite")

st.markdown("""
This app implements the PNVIKOR method using Pythagorean Neutrosophic Sets and Hamming distance to support MCDM.
""")

# Step 1: Upload decision matrix
df = None
st.header("Step 1: Upload Decision Matrix")
file = st.file_uploader("Upload your Excel file (.xlsx) with decision matrix", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    st.write("### Raw Decision Matrix", df)

    # Step 2: Input benefit/cost criteria
    st.header("Step 2: Define Criteria Type")
    criteria = df.columns[1:]
    criteria_type = {}
    for c in criteria:
        criteria_type[c] = st.selectbox(f"{c} is a", ["Benefit", "Cost"], key=c)

    # Step 3: Aggregate values - assuming single matrix already averaged
    st.header("Step 3: PNS Ratings Preview")
    st.markdown("Each entry must be a PNS triple: (τ, ξ, η)")

    def parse_pns(cell):
        try:
            return eval(cell) if isinstance(cell, str) else cell
        except:
            return (0, 0, 0)

    pns_matrix = df.iloc[:, 1:].applymap(parse_pns)

    # Step 4: Normalize decision matrix
    st.header("Step 4: Normalization")
    norm_matrix = []
    for j, c in enumerate(criteria):
        col = pns_matrix[c]
        tau = np.array([v[0] for v in col])
        xi = np.array([v[1] for v in col])
        eta = np.array([v[2] for v in col])

        if criteria_type[c] == "Benefit":
            tau_n = tau / tau.max()
            xi_n = xi / xi.max()
            eta_n = eta / eta.max()
        else:
            tau_n = tau.min() / tau
            xi_n = xi.min() / xi
            eta_n = eta.min() / eta

        norm_matrix.append(list(zip(tau_n, xi_n, eta_n)))

    norm_matrix = pd.DataFrame(norm_matrix).T
    norm_matrix.columns = criteria
    norm_matrix.insert(0, df.columns[0], df.iloc[:, 0])
    st.write("### Normalized Matrix", norm_matrix)

    # Step 5: Determine best/worst
    st.header("Step 5: Best & Worst Values")
    best = {}
    worst = {}
    for c in criteria:
        tau = np.array([v[0] for v in norm_matrix[c]])
        xi = np.array([v[1] for v in norm_matrix[c]])
        eta = np.array([v[2] for v in norm_matrix[c]])
        if criteria_type[c] == "Benefit":
            best[c] = (tau.max(), xi.min(), eta.min())
            worst[c] = (tau.min(), xi.max(), eta.max())
        else:
            best[c] = (tau.min(), xi.max(), eta.max())
            worst[c] = (tau.max(), xi.min(), eta.min())

    # Step 6: Calculate Si and Ri using Hamming distance
    st.header("Step 6: Utility (Si) and Regret (Ri) using Hamming Distance")

    def hamming(p1, p2):
        return (abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])) / 3

    Si = []
    Ri = []
    for i in range(len(norm_matrix)):
        s = 0
        r = -1
        for c in criteria:
            d = hamming(norm_matrix[c][i], best[c])
            s += d
            r = max(r, d)
        Si.append(s)
        Ri.append(r)

    st.write("Si (utility measure)", Si)
    st.write("Ri (regret measure)", Ri)

    # Step 7: Compute Qi
    st.header("Step 7: VIKOR Index Qi")
    v = st.slider("Select V value (strategy weight)", 0.0, 1.0, 0.5)
    S_star = min(Si)
    S_minus = max(Si)
    R_star = min(Ri)
    R_minus = max(Ri)

    Qi = [
        v * (s - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (r - R_star) / (R_minus - R_star + 1e-9)
        for s, r in zip(Si, Ri)
    ]

    result = pd.DataFrame({
        "Alternative": df.iloc[:, 0],
        "Si": Si,
        "Ri": Ri,
        "Qi": Qi
    }).sort_values("Qi")

    st.write("### Final Ranking", result.reset_index(drop=True))

    # Step 8: Compromise solution check
    st.header("Step 8: Compromise Solution")
    DQ = 1 / (len(result) - 1)
    Q1, Q2 = result["Qi"].iloc[0], result["Qi"].iloc[1]
    A1 = result["Alternative"].iloc[0]
    A2 = result["Alternative"].iloc[1]
    S1 = result["Si"].iloc[0]
    R1 = result["Ri"].iloc[0]

    cond1 = (Q2 - Q1) >= DQ
    cond2 = (result.sort_values("Si")["Alternative"].iloc[0] == A1 or result.sort_values("Ri")["Alternative"].iloc[0] == A1)

    if cond1 and cond2:
        st.success(f"{A1} is the compromise solution.")
    elif cond2:
        st.info(f"Compromise set: {A1}, {A2}")
    else:
        Qlist = result[result["Qi"] - Q1 < DQ]["Alternative"].tolist()
        st.warning(f"Compromise set: {', '.join(Qlist)}")
