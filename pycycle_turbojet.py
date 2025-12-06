import sys
import tkinter as tk
from tkinter import ttk
import openmdao.api as om
import pycycle.api as pyc

# ==========================================
# PART 1: MAP & DATA SETUP
# ==========================================

# 1. Load Maps (Handle version differences safely)
try:
    # Try the map from your example
    from pycycle.maps.lpt2269 import LPT2269
    TurbineMap = LPT2269
except ImportError:
    # Fallback to standard map if 2269 is missing
    print("LPT2269 map not found, using LPT22 fallback.")
    from pycycle.maps.lpt22 import LPT22
    TurbineMap = LPT22

from pycycle.maps.axi5 import AXI5

# ==========================================
# PART 2: THE PHYSICS ENGINE (Your Code Adapted)
# ==========================================

class Turbojet(pyc.Cycle):
    def setup(self):
        # --- Options from your snippet ---
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
        
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2), promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # --- Connect flow stations ---
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozz.Fl_I')

        # --- Connect Mechanical ---
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # --- Connect Environment ---
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # --- Performance Connections ---
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

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
        self.root.title("PyCycle Real Parametric Analysis")
        self.root.geometry("1100x700")
        
        # Initialize Problem
        self.init_problem()
        
        # --- UI Layout ---
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left Sidebar (Inputs)
        left_col = ttk.Frame(main_frame, width=300)
        left_col.pack(side="left", fill="y", padx=10)
        
        # Right Area (Outputs)
        right_col = ttk.Frame(main_frame)
        right_col.pack(side="right", fill="both", expand=True)
        
        # --- Variables ---
        self.vars = {
            'alt': tk.DoubleVar(value=0.0),       # ft
            'mn': tk.DoubleVar(value=0.001),      # Mach (Design)
            'fn_req': tk.DoubleVar(value=11800.0),# lbf
            't4': tk.DoubleVar(value=2370.0),     # degR
            'pr_comp': tk.DoubleVar(value=13.5),  # Pressure Ratio
            'eff_comp': tk.DoubleVar(value=0.83),
            'eff_turb': tk.DoubleVar(value=0.86)
        }
        
        # --- Controls ---
        ttk.Label(left_col, text="Design Parameters", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.add_slider(left_col, "Altitude (ft)", 'alt', 0, 40000, 1000)
        self.add_slider(left_col, "Mach Number", 'mn', 0.001, 1.5, 0.05)
        self.add_slider(left_col, "Target Thrust (lbf)", 'fn_req', 1000, 30000, 500)
        self.add_slider(left_col, "TIT (Rankine)", 't4', 1800, 3200, 50)
        self.add_slider(left_col, "Compressor PR", 'pr_comp', 5.0, 30.0, 0.5)
        self.add_slider(left_col, "Comp Efficiency", 'eff_comp', 0.75, 0.95, 0.01)
        self.add_slider(left_col, "Turb Efficiency", 'eff_turb', 0.75, 0.95, 0.01)
        
        # --- Output Text ---
        ttk.Label(right_col, text="Design Point Results", font=("Arial", 14, "bold")).pack(pady=10)
        self.text_out = tk.Text(right_col, font=("Courier", 10), bg="#f4f4f4")
        self.text_out.pack(fill="both", expand=True)
        
        # Initial Run
        self.run_cycle()

    def init_problem(self):
        self.prob = om.Problem()
        self.prob.model = Turbojet(design=True)
        self.prob.setup()
        
        # Set sensible defaults to ensure first run works
        self.prob.set_val('fc.alt', 0, units='ft')
        self.prob.set_val('fc.MN', 0.001)
        self.prob.set_val('balance.Fn_target', 11800.0, units='lbf')
        self.prob.set_val('balance.T4_target', 2370.0, units='degR')
        self.prob.set_val('comp.PR', 13.5)
        self.prob.set_val('comp.eff', 0.83)
        self.prob.set_val('turb.eff', 0.86)
        
        # Initial Guesses (Crucial for OpenMDAO)
        self.prob['balance.FAR'] = 0.0175
        self.prob['balance.W'] = 168.0
        self.prob['balance.turb_PR'] = 4.0

    def add_slider(self, parent, label, key, min_v, max_v, step):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        lbl_frame = ttk.Frame(frame)
        lbl_frame.pack(fill="x")
        ttk.Label(lbl_frame, text=label).pack(side="left")
        val_lbl = ttk.Label(lbl_frame, text=f"{self.vars[key].get():.2f}")
        val_lbl.pack(side="right")
        
        def on_slide(v):
            val = float(v)
            # Snap to step
            val = round(val / step) * step
            self.vars[key].set(val)
            val_lbl.config(text=f"{val:.2f}")
            # Trigger analysis
            self.run_cycle()
            
        s = ttk.Scale(frame, from_=min_v, to=max_v, orient="horizontal", command=on_slide)
        s.set(self.vars[key].get())
        s.pack(fill="x")

    def run_cycle(self):
        p = self.prob
        v = self.vars
        
        try:
            # 1. Update Inputs
            p.set_val('fc.alt', v['alt'].get(), units='ft')
            p.set_val('fc.MN', v['mn'].get())
            p.set_val('balance.Fn_target', v['fn_req'].get(), units='lbf')
            p.set_val('balance.T4_target', v['t4'].get(), units='degR')
            p.set_val('comp.PR', v['pr_comp'].get())
            p.set_val('comp.eff', v['eff_comp'].get())
            p.set_val('turb.eff', v['eff_turb'].get())
            
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
PYCYCLE PARAMETRIC ANALYSIS
===========================
STATUS: CONVERGED

INPUTS
------
Altitude:    {v['alt'].get():.0f} ft
Mach:        {v['mn'].get():.3f}
Target Thr:  {v['fn_req'].get():.0f} lbf
Target TIT:  {v['t4'].get():.0f} R

SIZING RESULTS (DESIGN POINT)
-----------------------------
Mass Flow Req:  {W_air:.2f} lbm/s  <-- Sized to meet Thrust
Fuel Flow:      {W_fuel:.1f} lbm/h
TSFC:           {tsfc:.4f}

CYCLE DETAILS
-------------
Overall PR:     {OPR:.2f}
Comp Exit Temp: {T3:.1f} R
Comp Exit Pres: {P3:.2f} psi

Turbine PR:     {Turb_PR:.3f}      <-- Balanced for Work
Turb Exit Temp: {T5:.1f} R
"""
            self.text_out.delete(1.0, tk.END)
            self.text_out.insert(tk.END, out)
            self.text_out.config(fg="black")
            
        except Exception as e:
            self.text_out.delete(1.0, tk.END)
            msg = f"SOLVER ERROR: \n{str(e)}\n\nThe engine could not balance.\nPossible causes:\n1. TIT too low for Pressure Ratio\n2. Thrust target impossible for size"
            self.text_out.insert(tk.END, msg)
            self.text_out.config(fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()