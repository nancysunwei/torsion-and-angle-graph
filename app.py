import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Set page title and layout
st.set_page_config(page_title="Torsion Analysis Module", layout="wide")

st.title("Torsion Analysis Module")
st.markdown("Convert Tkinter desktop torsion analysis module to Streamlit web app. Please enter parameters in the table below and click compute.")

# --- Initialize Session State variables ---
if 'n_elem_t' not in st.session_state:
    st.session_state['n_elem_t'] = 3
if 'unit_system_t' not in st.session_state:
    st.session_state['unit_system_t'] = 'SI (mm/N·m)'

# --- Sidebar: Global settings and units ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Preset unit system toggle
    st.subheader("Presets")
    col1, col2 = st.columns(2)
    if col1.button("US (in/lb·in)"):
        st.session_state['unit_system_t'] = 'US (in/lb·in)'
    if col2.button("SI (mm/N·m)"):
        st.session_state['unit_system_t'] = 'SI (mm/N·m)'

    # Number of elements
    n_elem = st.number_input("Shaft Segments:", min_value=1, max_value=20, value=st.session_state['n_elem_t'], step=1)
    st.session_state['n_elem_t'] = n_elem

    # Unit dropdowns
    st.subheader("Units")
    is_si = st.session_state['unit_system_t'] == 'SI (mm/N·m)'
    
    u_len = st.selectbox("Length", ["mm", "m", "in", "ft"], index=0 if is_si else 2)
    u_diam = st.selectbox("Diameter", ["mm", "m", "in"], index=0 if is_si else 2)
    u_mod = st.selectbox("Shear Modulus (G)", ["GPa", "MPa", "Pa", "psi", "ksi"], index=0 if is_si else 3)
    u_torque = st.selectbox("Torque", ["N·m", "kN·m", "lb·in", "lb·ft"], index=0 if is_si else 2)
    u_angle = st.selectbox("Angle", ["rad", "deg"], index=0)

# --- Generate default data tables ---
# Provide different default values based on the unit system
def_L = 100.0 if is_si else 10.0
def_OD = 50.0 if is_si else 2.0
def_ID = 0.0 # Solid shaft by default
def_G = 80.0 if is_si else 11000.0 # G for steel: ~80GPa or ~11000ksi

n_node = n_elem + 1

# Create Element DataFrame
elem_data = {
    f"Length L ({u_len})": [def_L] * n_elem,
    f"Outer Dia. OD ({u_diam})": [def_OD] * n_elem,
    f"Inner Dia. ID ({u_diam})": [def_ID] * n_elem,
    f"Shear Modulus G ({u_mod})": [def_G] * n_elem
}
df_elem_default = pd.DataFrame(elem_data)
df_elem_default.index = [f"Elem {i+1}" for i in range(n_elem)]

# Create Node DataFrame
node_data = {
    f"Ext. Torque T ({u_torque})": [0.0] * n_node,
    "Constraint (1=Fix, 0=Free)": [1 if i == 0 else 0 for i in range(n_node)]
}
df_node_default = pd.DataFrame(node_data)
df_node_default.index = [f"Node {i+1}" for i in range(n_node)]

# --- Main Screen: Data Input Area ---
st.subheader("📝 Input Data")
st.caption("You can directly edit the values in the tables below")

col_table1, col_table2 = st.columns([1.5, 1])

with col_table1:
    st.markdown("**Shaft Properties**")
    df_elem = st.data_editor(df_elem_default, use_container_width=True)

with col_table2:
    st.markdown("**Nodal Torques & BCs**")
    df_node = st.data_editor(df_node_default, use_container_width=True)

# --- Calculation & Plotting ---
st.markdown("---")
if st.button("🚀 Compute & Plot", type="primary"):
    try:
        # Extract data
        L = df_elem.iloc[:, 0].values.astype(float)
        OD = df_elem.iloc[:, 1].values.astype(float)
        ID = df_elem.iloc[:, 2].values.astype(float)
        G = df_elem.iloc[:, 3].values.astype(float)
        
        Torques = df_node.iloc[:, 0].values.astype(float)
        Constraints = df_node.iloc[:, 1].values.astype(int)

        # Check validity of inner/outer diameters
        if any(ID >= OD):
            st.error("Error: Inner Diameter (ID) must be less than Outer Diameter (OD).")
            st.stop()

        # Calculate polar moment of inertia J
        J = [math.pi * (od**4 - id**4) / 32.0 for od, id in zip(OD, ID)]

        # --- Stiffness Matrix Calculation ---
        K = np.zeros((n_node, n_node))
        F = np.array(Torques)

        for i in range(n_elem):
            # Torsional stiffness k = GJ / L
            k = (G[i] * J[i]) / L[i]
            K[i, i]     += k
            K[i, i+1]   -= k
            K[i+1, i]   -= k
            K[i+1, i+1] += k

        # --- Boundary Conditions (Penalty Method) ---
        penalty = 1e20
        for i in range(n_node):
            if Constraints[i] == 1:
                K[i, i] *= penalty
                F[i] = 0

        # --- Solve for Twist Angle Phi ---
        Phi = np.linalg.solve(K, F)
        
        # --- Calculate Internal Torque ---
        Internal_T = []
        for i in range(n_elem):
            t_val = ((G[i] * J[i]) / L[i]) * (Phi[i+1] - Phi[i])
            Internal_T.append(t_val)

        # Unit conversion for angle display
        if u_angle == "deg":
            Phi_Display = np.degrees(Phi)
        else:
            Phi_Display = Phi

        # --- Find Extremes ---
        abs_t = [abs(t) for t in Internal_T]
        max_t_idx = np.argmax(abs_t)
        max_t_val = Internal_T[max_t_idx]
        
        abs_phi = [abs(p) for p in Phi_Display]
        max_phi_idx = np.argmax(abs_phi)
        max_phi_val = Phi_Display[max_phi_idx]

        # --- Display Critical Results ---
        st.subheader("📊 Critical Results Summary")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(label=f"Max Torque @ Elem {max_t_idx+1}", value=f"{max_t_val:.2f} {u_torque}")
        res_col2.metric(label=f"Max Twist Angle @ Node {max_phi_idx+1}", value=f"{max_phi_val:.4f} {u_angle}")

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_facecolor('#ffffff')
        plt.subplots_adjust(hspace=0.3)

        x_coords = [0]
        cur_x = 0
        for l in L: 
            cur_x += l
            x_coords.append(cur_x)

        # 1. Torque Diagram
        x_plot, y_plot = [], []
        for i in range(n_elem):
            x_plot.extend([x_coords[i], x_coords[i+1]])
            y_plot.extend([Internal_T[i], Internal_T[i]])

        color_t = '#673ab7' # Deep Purple
        ax1.plot(x_plot, y_plot, color=color_t, linewidth=2)
        ax1.fill_between(x_plot, y_plot, 0, alpha=0.2, color=color_t)
        ax1.set_title("Internal Torque Diagram", fontsize=12, fontweight='bold')
        ax1.set_ylabel(f"Torque ({u_torque})", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        y_min, y_max = min(y_plot), max(y_plot)
        margin = (y_max - y_min) * 0.4 if y_max != y_min else abs(y_max)*0.5 + 1.0
        ax1.set_ylim(y_min - margin*0.5, y_max + margin)

        # Annotate Torque
        for i in range(n_elem):
            mid_x = (x_coords[i] + x_coords[i+1]) / 2
            val = Internal_T[i]
            is_max = (i == max_t_idx)
            
            if is_max:
                offset_y = margin * 0.8
                ax1.annotate(f"MAX: {val:.2f}", 
                             xy=(mid_x, val), 
                             xytext=(mid_x, val + offset_y),
                             arrowprops=dict(facecolor='red', arrowstyle='->', connectionstyle="arc3"),
                             fontsize=10, color='red', fontweight='bold', ha='center',
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2'))
            else:
                force_range = max(Internal_T) - min(Internal_T)
                offset = force_range * 0.05 if force_range != 0 else 1.0
                ax1.text(mid_x, val + offset, f"{val:.2f}", 
                        ha='center', va='bottom', 
                        fontsize=9, color=color_t,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        # 2. Angle of Twist Diagram
        color_phi = '#e91e63' # Pink
        ax2.plot(x_coords, Phi_Display, color=color_phi, marker='o', linewidth=2, markersize=6)
        ax2.set_title("Angle of Twist Diagram", fontsize=12, fontweight='bold')
        ax2.set_ylabel(f"Angle ({u_angle})", fontsize=10)
        ax2.set_xlabel(f"Position ({u_len})", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        y_d_min, y_d_max = min(Phi_Display), max(Phi_Display)
        margin_d = (y_d_max - y_d_min) * 0.3 if y_d_max != y_d_min else abs(y_d_max)*0.5 + 0.1
        ax2.set_ylim(y_d_min - margin_d, y_d_max + margin_d)

        # Annotate Twist Angle
        for i, (x, y) in enumerate(zip(x_coords, Phi_Display)):
            is_max = (i == max_phi_idx)
            
            if is_max:
                arrow_offset = 40 if y >= 0 else -40 
                ax2.annotate(f"MAX: {y:.4f}", 
                             xy=(x, y), 
                             xytext=(0, arrow_offset), 
                             textcoords='offset points',
                             arrowprops=dict(facecolor='red', arrowstyle='->'),
                             fontsize=10, color='red', fontweight='bold', ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.2'))
            else:
                y_offset = 15 if i % 2 == 0 else -20
                ax2.annotate(f"{y:.4f}", 
                             xy=(x, y), 
                             xytext=(0, y_offset), 
                             textcoords='offset points',
                             ha='center', va='center',
                             fontsize=9, color=color_phi,
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during calculation: {e}")
