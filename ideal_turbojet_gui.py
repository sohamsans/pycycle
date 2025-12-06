import tkinter as tk
from tkinter import ttk, messagebox
import openmdao.api as om
import pycycle.api as pyc
import sys

# --- PYCYCLE TURBOJET MODEL CLASS ---
class TurbojetModel(pyc.Cycle):
    def initialize(self):
        self.options.declare('thermo_method', default='CEA', values=['CEA', 'TABULAR'])
        self.options.declare('thermo_data', default=None)
        super().initialize()

    def setup(self):
        design = True
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        
        # 1. Add Components
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet(design=design))
        
        # Compressor (Uses standard HPC Map as placeholder for design point)
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.HPCMap, design=design, 
                                                  map_extrap=True))
        
        self.add_subsystem('burner', pyc.Combustor(design=design, fuel_type='Jet-A(g)'))
        
        # Turbine (Uses standard HPT Map as placeholder)
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.HPTMap, design=design, 
                                               map_extrap=True))
        
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CD', lossCoef='Cv', design=design))
        
        # Shaft (Connects Compressor and Turbine)
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2))

        # 2. Add Solver/Balances
        balance = self.add_subsystem('balance', om.BalanceComp())
        
        # Balance A: Fuel-Air Ratio (FAR) to match Target T4
        balance.add_balance('FAR', val=0.02, units=None, eq_units='degR', lower=0.001, upper=0.2)
        
        # Balance B: Turbine Pressure Ratio (PR) to match Shaft Power
        balance.add_balance('turb_PR', val=2.0, lower=1.001, upper=20.0, eq_units='hp', use_mult=True, mult_val=-1)

        # 3. Connect Flow Path
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozzle.Fl_I')
        
        # 4. Connect Mechanical (Shaft)
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # 5. Connect Solver/Balances
        
        # T4 Control
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
        
        # Shaft Power Balance
        self.connect('balance.turb_PR', 'turb.PR')
        self.connect('shaft.pwr_in_real', 'balance.lhs:turb_PR')  # Power req by Comp
        self.connect('shaft.pwr_out_real', 'balance.rhs:turb_PR') # Power gen by Turb

        # 6. Cycle Closure
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        super().setup()

class TurbojetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyCycle Turbojet Simulator")
        self.root.geometry("600x850") # Increased height for more options
        self.root.configure(bg="#f0f0f0")

        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background="#f0f0f0", font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), foreground="#333")
        style.configure('SubHeader.TLabel', font=('Arial', 11, 'bold', 'underline'), foreground="#555")

        # Header
        ttk.Label(root, text="Parametric Turbojet Cycle", style='Header.TLabel').pack(pady=15)

        # --- Main Input Frame ---
        input_frame = ttk.LabelFrame(root, text="Cycle Inputs", padding="15")
        input_frame.pack(fill=tk.BOTH, expand=False, padx=20)

        # Design Variables
        self.mach_var = tk.DoubleVar(value=0.8)
        self.alt_var = tk.DoubleVar(value=35000.0)
        self.cpr_var = tk.DoubleVar(value=20.0)
        self.t4_var = tk.DoubleVar(value=3000.0)
        
        # Efficiency/Loss Variables (Defaults set to Ideal)
        self.comp_eff_var = tk.DoubleVar(value=1.0)
        self.turb_eff_var = tk.DoubleVar(value=1.0)
        self.inlet_rec_var = tk.DoubleVar(value=1.0)
        self.nozz_cv_var = tk.DoubleVar(value=1.0)
        self.burner_loss_var = tk.DoubleVar(value=0.0)
        self.shaft_loss_var = tk.DoubleVar(value=0.0)

        # Layout - Design Points
        row = 0
        ttk.Label(input_frame, text="Design Point", style='SubHeader.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(0,5), columnspan=2)
        row += 1
        self.create_input(input_frame, "Mach Number:", self.mach_var, row)
        row += 1
        self.create_input(input_frame, "Altitude (ft):", self.alt_var, row)
        row += 1
        self.create_input(input_frame, "Compressor Pressure Ratio:", self.cpr_var, row)
        row += 1
        self.create_input(input_frame, "Burner Exit Temp (degR):", self.t4_var, row)
        
        # Layout - Component Efficiencies
        row += 1
        ttk.Label(input_frame, text="Component Quality (1.0 = Ideal)", style='SubHeader.TLabel').grid(row=row, column=0, sticky=tk.W, pady=(15,5), columnspan=2)
        row += 1
        self.create_input(input_frame, "Inlet Recovery (0-1):", self.inlet_rec_var, row)
        row += 1
        self.create_input(input_frame, "Compressor Poly Eff (0-1):", self.comp_eff_var, row)
        row += 1
        self.create_input(input_frame, "Burner Press Loss (dP/P):", self.burner_loss_var, row)
        row += 1
        self.create_input(input_frame, "Turbine Poly Eff (0-1):", self.turb_eff_var, row)
        row += 1
        self.create_input(input_frame, "Nozzle Velocity Coeff:", self.nozz_cv_var, row)
        row += 1
        self.create_input(input_frame, "Shaft Mech Loss (frac):", self.shaft_loss_var, row)
        
        # Run Button
        ttk.Button(root, text="Run Simulation", command=self.run_simulation).pack(pady=15)

        # Output Frame
        self.output_text = tk.Text(root, height=18, width=70, state='disabled', font=('Consolas', 10))
        self.output_text.pack(padx=20, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def create_input(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=variable, width=15).grid(row=row, column=1, sticky=tk.E, pady=2, padx=10)

    def run_simulation(self):
        self.status_var.set("Running...")
        self.root.update()

        try:
            # 1. Setup OpenMDAO Problem
            prob = om.Problem()
            prob.model = TurbojetModel(thermo_method='CEA', thermo_data=pyc.species_data.janaf)
            
            # Setup solvers
            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.options['iprint'] = 0 
            prob.model.nonlinear_solver.options['maxiter'] = 50
            prob.model.linear_solver = om.DirectSolver()

            prob.setup(check=False)

            # 2. Set Inputs
            prob.set_val('fc.alt', self.alt_var.get(), units='ft')
            prob.set_val('fc.MN', self.mach_var.get())
            prob.set_val('comp.PR', self.cpr_var.get())
            
            # Set Target T4
            prob.set_val('balance.rhs:FAR', self.t4_var.get(), units='degR')

            # --- SET COMPONENT EFFICIENCIES FROM GUI ---
            prob.set_val('inlet.ram_recovery', self.inlet_rec_var.get())
            prob.set_val('comp.eff', self.comp_eff_var.get())       
            prob.set_val('burner.dPqP', self.burner_loss_var.get())    
            prob.set_val('turb.eff', self.turb_eff_var.get())       
            prob.set_val('nozzle.Cv', self.nozz_cv_var.get())      
            prob.set_val('shaft.fracLoss', self.shaft_loss_var.get()) 
            
            # 3. Run
            prob.run_model()

            # 4. Extract Results
            # Environment
            p0 = prob.get_val('fc.Fl_O:stat:P', units='psi')[0]
            
            # Pressures/Temps
            p2 = prob.get_val('inlet.Fl_O:tot:P', units='psi')[0]
            p3 = prob.get_val('comp.Fl_O:tot:P', units='psi')[0]
            t3 = prob.get_val('comp.Fl_O:tot:T', units='degR')[0]
            t4 = prob.get_val('burner.Fl_O:tot:T', units='degR')[0]
            p4 = prob.get_val('burner.Fl_O:tot:P', units='psi')[0]
            p5 = prob.get_val('turb.Fl_O:tot:P', units='psi')[0]
            t5 = prob.get_val('turb.Fl_O:tot:T', units='degR')[0]
            
            # Performance
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
            res += f"-------------------------\n"
            res += f"Compressor PR:        {self.cpr_var.get():.2f}\n"
            res += f"Compressor Efficiency:{self.comp_eff_var.get():.2f}\n"
            res += f"Compressor Exit P3:   {p3:.2f} psi\n"
            res += f"Compressor Exit T3:   {t3:.2f} degR\n"
            res += f"-------------------------\n"
            res += f"Turbine Inlet T4:     {t4:.2f} degR\n"
            res += f"Turbine Exit T5:      {t5:.2f} degR\n"
            res += f"Turbine Exit P5:      {p5:.2f} psi\n"
            res += f"Turbine PR:           {prob.get_val('turb.PR')[0]:.3f}\n"
            res += f"-------------------------\n"
            res += f"Fuel-Air Ratio:       {far:.5f}\n"
            res += f"Mass Flow:            {m_dot:.2f} lbm/s\n"
            
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
    app = TurbojetApp(root)
    root.mainloop()