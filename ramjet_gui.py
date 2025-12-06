import tkinter as tk
from tkinter import ttk, messagebox
import openmdao.api as om
import pycycle.api as pyc
import sys

# --- PYCYCLE RAMJET MODEL CLASS ---
class RamjetModel(pyc.Cycle):
    def initialize(self):
        # Initialize the cycle options
        self.options.declare('thermo_method', default='CEA', values=['CEA', 'TABULAR'])
        self.options.declare('thermo_data', default=None)
        super().initialize()

    def setup(self):
        # Design Point Analysis
        design = True 
        
        # 1. Add Components
        
        # Flight Conditions
        self.add_subsystem('fc', pyc.FlightConditions())
        
        # Inlet (Ideal)
        self.add_subsystem('inlet', pyc.Inlet(design=design))
        
        # Burner (Combustor)
        self.add_subsystem('burner', pyc.Combustor(design=design, fuel_type='Jet-A(g)'))
        
        # Nozzle (Ideal CD Nozzle)
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CD', 
                                                lossCoef='Cv', 
                                                design=design))

        # 2. Add Solver/Balance for T4 Control
        # FIX: We vary Fuel-Air Ratio (FAR) to achieve the target T4.
        # The Combustor takes FAR as input and calculates resulting Wfuel and T4.
        balance = self.add_subsystem('balance', om.BalanceComp())
        
        # Balance config: Vary 'FAR' (unitless) until LHS (Temp) equals RHS (Target Temp)
        balance.add_balance('FAR', val=0.02, units=None, eq_units='degR', lower=0.001, upper=0.2)
        
        # 3. Connect Components (The Flow Path)
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'nozzle.Fl_I')

        # 4. Connect Solver
        # Connect Balance Output (FAR) to Burner Input (Fuel-Air Ratio)
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        
        # Connect Burner Exit Temp to Balance LHS (The value we are matching against RHS)
        self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

        # 5. Cycle Closure
        # For ideal ramjet, P_exit = P_ambient (Ideal expansion)
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        # CRITICAL: Call super().setup() at the END.
        super().setup()

class RamjetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyCycle Ideal Ramjet Simulator")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background="#f0f0f0", font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), foreground="#333")

        # Header
        ttk.Label(root, text="Ideal Parametric Ramjet", style='Header.TLabel').pack(pady=15)

        # Input Frame
        input_frame = ttk.LabelFrame(root, text="Design Inputs", padding="20")
        input_frame.pack(fill=tk.BOTH, expand=False, padx=20)

        # Variables
        self.mach_var = tk.DoubleVar(value=3.0)
        self.alt_var = tk.DoubleVar(value=30000.0) # ft
        self.t4_var = tk.DoubleVar(value=3500.0)   # Rankine
        
        # Inputs
        self.create_input(input_frame, "Mach Number:", self.mach_var, 0)
        self.create_input(input_frame, "Altitude (ft):", self.alt_var, 1)
        self.create_input(input_frame, "Burner Exit Temp (degR):", self.t4_var, 2)
        
        # Run Button
        ttk.Button(root, text="Run Simulation", command=self.run_simulation).pack(pady=15)

        # Output Frame
        self.output_text = tk.Text(root, height=20, width=65, state='disabled', font=('Consolas', 10))
        self.output_text.pack(padx=20, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def create_input(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=variable, width=15).grid(row=row, column=1, sticky=tk.E, pady=5, padx=10)

    def run_simulation(self):
        self.status_var.set("Running...")
        self.root.update()

        try:
            # 1. Setup OpenMDAO Problem
            prob = om.Problem()
            # Use JANAF thermo data as in the HBTF example
            prob.model = RamjetModel(thermo_method='CEA', thermo_data=pyc.species_data.janaf)
            
            # Setup solvers
            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.options['iprint'] = 0 
            prob.model.nonlinear_solver.options['maxiter'] = 20
            prob.model.linear_solver = om.DirectSolver()

            prob.setup(check=False)

            # 2. Set Inputs
            prob.set_val('fc.alt', self.alt_var.get(), units='ft')
            prob.set_val('fc.MN', self.mach_var.get())
            
            # Set Target T4 on the Balance RHS (Using 'FAR' balance name now)
            prob.set_val('balance.rhs:FAR', self.t4_var.get(), units='degR')

            # Set Ideal Component Parameters
            prob.set_val('inlet.ram_recovery', 1.0) 
            prob.set_val('nozzle.Cv', 1.0)          
            
            # 3. Run
            prob.run_model()

            # 4. Extract Results
            p0 = prob.get_val('fc.Fl_O:stat:P', units='psi')[0]
            p3 = prob.get_val('burner.Fl_I:tot:P', units='psi')[0]
            t3 = prob.get_val('burner.Fl_I:tot:T', units='degR')[0]
            t4 = prob.get_val('burner.Fl_O:tot:T', units='degR')[0]
            
            # 'phi' might not be readily available if we drive FAR directly, 
            # but we can get FAR (Fuel-Air Ratio)
            far = prob.get_val('balance.FAR')[0]
            
            fg = prob.get_val('nozzle.Fg', units='lbf')[0]
            
            m_dot = prob.get_val('inlet.Fl_O:stat:W', units='lbm/s')[0]
            v_flight = prob.get_val('fc.Fl_O:stat:V', units='ft/s')[0]
            fram = (m_dot * v_flight) / 32.174 # lbf conversion
            
            fn = fg - fram
            w_fuel = prob.get_val('burner.Wfuel', units='lbm/h')[0]
            tsfc = w_fuel / fn if fn > 0 else 0.0
            isp = fn / (w_fuel / 3600.0) if w_fuel > 0 else 0.0

            # 5. Display
            res = f"--- SIMULATION RESULTS ---\n"
            res += f"Net Thrust (Fn):      {fn:.2f} lbf\n"
            res += f"TSFC:                 {tsfc:.4f} (lbm/h)/lbf\n"
            res += f"Specific Impulse:     {isp:.2f} s\n"
            res += f"Ram Drag:             {fram:.2f} lbf\n"
            res += f"-------------------------\n"
            res += f"Air Mass Flow:        {m_dot:.2f} lbm/s\n"
            res += f"Fuel-Air Ratio:       {far:.5f}\n"
            res += f"Burner Inlet P3:      {p3:.2f} psi\n"
            res += f"Burner Inlet T3:      {t3:.2f} degR\n"
            res += f"Burner Exit T4:       {t4:.2f} degR\n"
            res += f"-------------------------\n"
            res += f"Flight Speed:         {v_flight:.1f} ft/s\n"
            res += f"Ambient Pressure:     {p0:.3f} psi\n"
            
            self.display_output(res)
            self.status_var.set("Success")

        except Exception as e:
            import traceback
            err_msg = f"Error running simulation:\n{str(e)}\n\n{traceback.format_exc()}"
            self.display_output(err_msg)
            self.status_var.set("Error")

    def display_output(self, text):
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = RamjetApp(root)
    root.mainloop()