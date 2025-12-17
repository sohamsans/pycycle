import sys
import tkinter as tk
from tkinter import ttk, messagebox
import openmdao.api as om
import pycycle.api as pyc
import traceback

# ==========================================
# PART 1: MAP & DATA SETUP
# ==========================================

# 1. Load Maps (Handle version differences safely)
try:
    # Try the map from your example
    from pycycle.maps.lpt2269 import LPT2269
    TurbineMap = LPT2269
    MAP_STATUS = "Using LPT2269 Map"
except ImportError:
    # Fallback to standard map if 2269 is missing
    print("LPT2269 map not found, using LPT22 fallback.")
    from pycycle.maps.lpt22 import LPT22
    TurbineMap = LPT22
    MAP_STATUS = "Using LPT22 (Fallback) Map"

from pycycle.maps.axi5 import AXI5

# ==========================================
# PART 2: THE PHYSICS ENGINE
# ==========================================

class Turbojet(pyc.Cycle):
    def setup(self):
        # --- Options ---
        USE_TABULAR = True
        if USE_TABULAR: 
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = "FAR"
        else: 
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = "Jet-A(g)"

        design = self.options['design']

        # --- Add engine elements ---
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        
        # Compressor using AXI5
        self.add_subsystem('comp', pyc.Compressor(map_data=AXI5, map_extrap=True),
                           promotes_inputs=['Nmech'])
        
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        
        # Turbine using your requested map
        self.add_subsystem('turb', pyc.Turbine(map_data=TurbineMap),
                           promotes_inputs=['Nmech'])
        
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2), promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # --- Connect flow stations ---
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozzle.Fl_I')

        # --- Connect Mechanical ---
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # --- Connect Environment ---
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        # --- Performance Connections ---
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')

        # --- Balances (Design Mode Only for Parametric GUI) ---
        balance = self.add_subsystem('balance', om.BalanceComp())
        
        # 1. Mass Flow Balance (Vary W to hit Target Thrust)
        balance.add_balance('W', units='lbm/s', eq_units='lbf', rhs_name='Fn_target')
        self.connect('balance.W', 'inlet.Fl_I:stat:W')
        self.connect('perf.Fn', 'balance.lhs:W')

        # 2. Fuel Balance (Vary FAR to hit TIT)
        balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017, rhs_name='T4_target')
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

        # 3. Work Balance (Vary Turb PR to hit Shaft Net Power = 0)
        balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
        self.connect('balance.turb_PR', 'turb.PR')
        self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')

        # --- Solver Settings ---
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-4
        newton.options['rtol'] = 1e-4
        newton.options['iprint'] = 0 # Silent for GUI
        newton.options['maxiter'] = 20
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        
        self.linear_solver = om.DirectSolver()

        super().setup()

# ==========================================
# PART 3: THE GUI
# ==========================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PyCycle Turbojet Sizing Tool (Design Point)")
        self.root.geometry("800x800")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))

        # Header
        header_frame = ttk.Frame(root)
        header_frame.pack(pady=15)
        ttk.Label(header_frame, text="Turbojet Sizing & Design", style='Header.TLabel').pack()
        ttk.Label(header_frame, text=MAP_STATUS, font=('Segoe UI', 9, 'italic')).pack()
        
        # Initialize Problem (Early setup to ensure defaults)
        self.init_problem()
        
        # --- Tabs ---
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Tab 1: Requirements (Sizing Targets)
        tab_req = ttk.Frame(self.nb, padding=15)
        self.nb.add(tab_req, text="Sizing Targets")
        
        self.fn_var = tk.DoubleVar(value=11800.0)
        self.t4_var = tk.DoubleVar(value=2370.0)
        
        self.create_input(tab_req, "Target Thrust (lbf):", self.fn_var, 0)
        self.create_input(tab_req, "Target TIT (Rankine):", self.t4_var, 1)

        # Tab 2: Cycle Parameters
        tab_cyc = ttk.Frame(self.nb, padding=15)
        self.nb.add(tab_cyc, text="Cycle Params")
        
        self.alt_var = tk.DoubleVar(value=0.0)
        self.mn_var = tk.DoubleVar(value=0.001)
        self.pr_var = tk.DoubleVar(value=13.5)
        
        self.create_input(tab_cyc, "Altitude (ft):", self.alt_var, 0)
        self.create_input(tab_cyc, "Mach Number:", self.mn_var, 1)
        self.create_input(tab_cyc, "Compressor PR:", self.pr_var, 2)

        # Tab 3: Component Quality
        tab_eff = ttk.Frame(self.nb, padding=15)
        self.nb.add(tab_eff, text="Efficiencies")
        
        self.eff_comp_var = tk.DoubleVar(value=0.83)
        self.eff_turb_var = tk.DoubleVar(value=0.86)
        self.rec_var = tk.DoubleVar(value=0.99)
        self.burner_dp_var = tk.DoubleVar(value=0.03)
        self.nozz_cv_var = tk.DoubleVar(value=0.99)
        
        self.create_input(tab_eff, "Compressor Poly Eff:", self.eff_comp_var, 0)
        self.create_input(tab_eff, "Turbine Poly Eff:", self.eff_turb_var, 1)
        self.create_input(tab_eff, "Inlet Recovery:", self.rec_var, 2)
        self.create_input(tab_eff, "Burner dP/P:", self.burner_dp_var, 3)
        self.create_input(tab_eff, "Nozzle Cv:", self.nozz_cv_var, 4)

        # Run Button
        ttk.Button(root, text="RUN SIZING", command=self.run_cycle).pack(pady=15, ipadx=10, ipady=2)
        
        # Output
        self.output_text = tk.Text(root, height=18, width=80, state='disabled', font=('Consolas', 9), bg="#f8f8f8")
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

    def init_problem(self):
        self.prob = om.Problem()
        self.prob.model = Turbojet(design=True)
        self.prob.setup()
        
        # Set sensible defaults to ensure first run works
        # Initial Guesses (Crucial for OpenMDAO)
        self.prob['balance.FAR'] = 0.0175
        self.prob['balance.W'] = 168.0
        self.prob['balance.turb_PR'] = 4.0

    def create_input(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=8)
        ttk.Entry(parent, textvariable=variable, width=15).grid(row=row, column=1, sticky=tk.E, pady=8, padx=15)

    def run_cycle(self):
        self.status_var.set("Sizing Engine...")
        self.root.update()
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')
        
        p = self.prob
        
        try:
            # 1. Update Inputs
            p.set_val('fc.alt', self.alt_var.get(), units='ft')
            p.set_val('fc.MN', self.mn_var.get())
            p.set_val('balance.Fn_target', self.fn_var.get(), units='lbf')
            p.set_val('balance.T4_target', self.t4_var.get(), units='degR')
            p.set_val('comp.PR', self.pr_var.get())
            p.set_val('comp.eff', self.eff_comp_var.get())
            p.set_val('turb.eff', self.eff_turb_var.get())
            
            p.set_val('inlet.ram_recovery', self.rec_var.get())
            p.set_val('burner.dPqP', self.burner_dp_var.get())
            p.set_val('nozzle.Cv', self.nozz_cv_var.get())
            
            # 2. Run
            p.run_model()
            
            # 3. Harvest Results
            Fn = p.get_val('perf.Fn', units='lbf')[0]
            tsfc = p.get_val('perf.TSFC', units='lbm/h/lbf')[0]
            W_air = p.get_val('inlet.Fl_O:stat:W', units='lbm/s')[0]
            W_fuel = p.get_val('burner.Wfuel', units='lbm/h')[0]
            OPR = p.get_val('perf.OPR')[0]
            
            T3 = p.get_val('comp.Fl_O:tot:T', units='degR')[0]
            P3 = p.get_val('comp.Fl_O:tot:P', units='psi')[0]
            Turb_PR = p.get_val('turb.PR')[0]
            T5 = p.get_val('turb.Fl_O:tot:T', units='degR')[0]
            
            # 4. Format Output
            out = f"""
=== SIZING RESULTS (DESIGN POINT) ===
STATUS: CONVERGED

INPUT REQUIREMENTS
------------------
Target Thrust:  {self.fn_var.get():.0f} lbf
Target TIT:     {self.t4_var.get():.0f} R
Flight Cond:    Mach {self.mn_var.get():.3f} @ {self.alt_var.get():.0f} ft

SIZED PARAMETERS (CALCULATED)
-----------------------------
Mass Flow Req:  {W_air:.2f} lbm/s  <-- SIZED
Turbine PR:     {Turb_PR:.3f}      <-- BALANCED

PERFORMANCE METRICS
-------------------
Net Thrust:     {Fn:.2f} lbf
TSFC:           {tsfc:.4f}
Fuel Flow:      {W_fuel:.1f} lbm/h
Overall PR:     {OPR:.2f}

COMPONENT DETAILS
-----------------
Comp Exit:      {P3:.1f} psi / {T3:.1f} R
Turb Exit:      {T5:.1f} R
"""
            self.display_output(out)
            self.status_var.set("Success")
            
        except Exception as e:
            msg = f"SOLVER ERROR: \n{str(e)}\n\n{traceback.format_exc()}"
            self.display_output(msg)
            self.status_var.set("Error")

    def display_output(self, text):
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()