import tkinter as tk
from tkinter import ttk, messagebox
import openmdao.api as om
import pycycle.api as pyc
import sys
import traceback

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
        self.root.title("PyCycle Parametric Turbojet Simulator")
        self.root.geometry("700x800")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))

        # Header
        ttk.Label(root, text="Parametric Turbojet Analysis", style='Header.TLabel').pack(pady=15)

        # Tabs
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        # --- Tab 1: Flight Conditions ---
        tab_flight = ttk.Frame(self.nb, padding=15)
        self.nb.add(tab_flight, text="Flight Data")

        self.mach_var = tk.DoubleVar(value=0.8)
        self.alt_var = tk.DoubleVar(value=35000.0)

        self.create_input(tab_flight, "Mach Number:", self.mach_var, 0)
        self.create_input(tab_flight, "Altitude (ft):", self.alt_var, 1)
        
        # --- Tab 2: Cycle Design ---
        tab_cycle = ttk.Frame(self.nb, padding=15)
        self.nb.add(tab_cycle, text="Cycle Design")

        self.cpr_var = tk.DoubleVar(value=20.0)
        self.t4_var = tk.DoubleVar(value=3000.0)

        self.create_input(tab_cycle, "Compressor Pressure Ratio:", self.cpr_var, 0)
        self.create_input(tab_cycle, "Burner Exit Temp (degR):", self.t4_var, 1)

        # --- Tab 3: Component Efficiencies ---
        tab_eff = ttk.Frame(self.nb, padding=15)
        self.nb.add(tab_eff, text="Efficiencies")

        self.inlet_rec_var = tk.DoubleVar(value=0.98)
        self.comp_eff_var = tk.DoubleVar(value=0.85)
        self.burner_loss_var = tk.DoubleVar(value=0.03)
        self.turb_eff_var = tk.DoubleVar(value=0.90)
        self.nozz_cv_var = tk.DoubleVar(value=0.98)
        self.shaft_loss_var = tk.DoubleVar(value=0.02)

        self.create_input(tab_eff, "Inlet Recovery (0-1):", self.inlet_rec_var, 0)
        self.create_input(tab_eff, "Compressor Poly Eff (0-1):", self.comp_eff_var, 1)
        self.create_input(tab_eff, "Burner Press Loss (dP/P):", self.burner_loss_var, 2)
        self.create_input(tab_eff, "Turbine Poly Eff (0-1):", self.turb_eff_var, 3)
        self.create_input(tab_eff, "Nozzle Velocity Coeff:", self.nozz_cv_var, 4)
        self.create_input(tab_eff, "Shaft Mech Loss (frac):", self.shaft_loss_var, 5)

        # Run Button
        ttk.Button(root, text="RUN SIMULATION", command=self.run_simulation).pack(pady=10, ipadx=10, ipady=2)

        # Output Frame
        self.output_text = tk.Text(root, height=18, width=75, state='disabled', font=('Consolas', 9), bg="#f8f8f8")
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

    def create_input(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=8)
        ttk.Entry(parent, textvariable=variable, width=15).grid(row=row, column=1, sticky=tk.E, pady=8, padx=15)

    def run_simulation(self):
        self.status_var.set("Running...")
        self.root.update()
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')

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
            res = f"=== SIMULATION RESULTS ===\n"
            res += f"Net Thrust (Fn):      {fn:.2f} lbf\n"
            res += f"TSFC:                 {tsfc:.4f} (lbm/h)/lbf\n"
            res += f"Specific Impulse:     {isp:.2f} s\n"
            res += f"--------------------------\n"
            res += f"Compressor PR:        {self.cpr_var.get():.2f}\n"
            res += f"Compressor Exit P3:   {p3:.2f} psi\n"
            res += f"Compressor Exit T3:   {t3:.2f} degR\n"
            res += f"--------------------------\n"
            res += f"Turbine Inlet T4:     {t4:.2f} degR\n"
            res += f"Turbine Exit T5:      {t5:.2f} degR\n"
            res += f"Turbine Exit P5:      {p5:.2f} psi\n"
            res += f"Turbine PR:           {prob.get_val('turb.PR')[0]:.3f}\n"
            res += f"--------------------------\n"
            res += f"Fuel-Air Ratio:       {far:.5f}\n"
            res += f"Mass Flow:            {m_dot:.2f} lbm/s\n"
            res += f"Flight Speed:         {v_flight:.1f} ft/s\n"
            
            self.display_output(res)
            self.status_var.set("Success")

        except Exception as e:
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