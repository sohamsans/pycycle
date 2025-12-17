import sys
import io
import tkinter as tk
from tkinter import ttk, messagebox
import openmdao.api as om
import pycycle.api as pyc

# ==========================================
# CORE LOGIC & CYCLE DEFINITION
# ==========================================

class Turbojet(pyc.Cycle):
    def setup(self):
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

        # Add engine elements
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, map_extrap=True),
                                    promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269),
                                    promotes_inputs=['Nmech'])
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2),promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect flow stations
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozzle.Fl_I')

        # Make other non-flow connections
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        # Connect outputs to perfomance element
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')

        # Add balances for design and off-design
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf', rhs_name='Fn_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017, rhs_name='T4_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')
        else:
            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=.3, rhs_name='Fn_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            balance.add_balance('W', val=168.0, units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.Throat:stat:area', 'balance.lhs:W')
       
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 0 # Controlled by top level
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        
        self.linear_solver = om.DirectSolver()

        super().setup()

class MPTurbojet(pyc.MPCycle):
    def setup(self):
        # Create design instance of model
        self.pyc_add_pnt('DESIGN', Turbojet())

        self.set_input_defaults('DESIGN.Nmech', 8070.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.60)
        self.set_input_defaults('DESIGN.comp.MN', 0.020)
        self.set_input_defaults('DESIGN.burner.MN', 0.020)
        self.set_input_defaults('DESIGN.turb.MN', 0.4)

        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('nozzle.Cv', 0.99)
        
        # Off-design conditions are added dynamically in the main run logic now,
        # or we can pre-define slots. For simplicity in this static class, 
        # we will define slots but their values will be set by the GUI.
        # Note: In a pure dynamic setup, we might rebuild this class or 
        # add points dynamically, but pycycle structure usually expects setup() to define topology.
        # We will keep the default 2 OD points structure for this example 
        # but allow the GUI to change their parameter values.
        
        self.od_pts = ['OD0', 'OD1']
        
        # Defaults (will be overwritten by GUI)
        self.od_MNs = [0.000001, 0.2]
        self.od_alts = [0.0, 5000]
        self.od_Fns =[11000.0, 8000.0]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, Turbojet(design=False))
            self.set_input_defaults(pt+'.fc.MN', val=self.od_MNs[i])
            self.set_input_defaults(pt+'.fc.alt', self.od_alts[i], units='ft')
            self.set_input_defaults(pt+'.balance.Fn_target', self.od_Fns[i], units='lbf')  

        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('nozzle.Throat:stat:area', 'balance.rhs:W')

        super().setup()

# ==========================================
# REPORTING UTILS
# ==========================================

def get_viewer_output(prob, pt):
    """
    Capture the viewer output to a string buffer.
    """
    output = io.StringIO()
    
    summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'], prob[pt+'.inlet.Fl_O:stat:W'], 
                    prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'],
                    prob[pt+'.perf.OPR'], prob[pt+'.perf.TSFC'])

    print(file=output)
    print("----------------------------------------------------------------------------", file=output)
    print("                              POINT:", pt, file=output)
    print("----------------------------------------------------------------------------", file=output)
    print("                       PERFORMANCE CHARACTERISTICS", file=output)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=output)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" %summary_data, file=output)

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'comp.Fl_O', 'burner.Fl_O',
                'turb.Fl_O', 'nozzle.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=output)

    comp_names = ['comp']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=output)

    pyc.print_burner(prob, [f'{pt}.burner'], file=output)

    turb_names = ['turb']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=output)

    noz_names = ['nozzle']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=output)

    shaft_names = ['shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=output)
    
    return output.getvalue()

# ==========================================
# GUI APPLICATION
# ==========================================

class SimpleTurbojetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Point Turbojet Cycle Analysis")
        self.root.geometry("1000x800")
        
        # Configure Grid Weight
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- Main Layout ---
        # Left Panel: Inputs
        input_panel = ttk.Frame(root, padding="10")
        input_panel.grid(row=0, column=0, sticky="nsew")
        
        # Right Panel: Output
        output_panel = ttk.Frame(root, padding="10")
        output_panel.grid(row=0, column=1, sticky="nsew")
        output_panel.columnconfigure(0, weight=1)
        output_panel.rowconfigure(1, weight=1)

        # --- DESIGN POINT INPUTS ---
        ttk.Label(input_panel, text="Design Point Parameters", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        self.design_vars = {
            'alt': tk.DoubleVar(value=0.0),
            'mn': tk.DoubleVar(value=0.000001),
            'fn': tk.DoubleVar(value=11800.0),
            't4': tk.DoubleVar(value=2370.0),
            'pr': tk.DoubleVar(value=13.5),
            'eff_comp': tk.DoubleVar(value=0.83),
            'eff_turb': tk.DoubleVar(value=0.86)
        }
        
        dp_frame = ttk.LabelFrame(input_panel, text="Design Point (DESIGN)", padding=10)
        dp_frame.pack(fill='x', pady=5)
        
        self.add_entry(dp_frame, "Altitude (ft):", self.design_vars['alt'])
        self.add_entry(dp_frame, "Mach Number:", self.design_vars['mn'])
        self.add_entry(dp_frame, "Target Thrust (lbf):", self.design_vars['fn'])
        self.add_entry(dp_frame, "Target T4 (R):", self.design_vars['t4'])
        self.add_entry(dp_frame, "Compressor PR:", self.design_vars['pr'])
        self.add_entry(dp_frame, "Compressor Eff:", self.design_vars['eff_comp'])
        self.add_entry(dp_frame, "Turbine Eff:", self.design_vars['eff_turb'])

        # --- OFF-DESIGN INPUTS ---
        ttk.Label(input_panel, text="Off-Design Points", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(20, 10))
        
        od_frame = ttk.LabelFrame(input_panel, text="Conditions", padding=10)
        od_frame.pack(fill='x', pady=5)
        
        # Headers
        headers = ["Point", "Alt (ft)", "Mach", "Thrust (lbf)"]
        for col, h in enumerate(headers):
            ttk.Label(od_frame, text=h, font=('Arial', 9, 'bold')).grid(row=0, column=col, padx=5, pady=2)
            
        self.od_vars = []
        # Pre-define 2 points matching the MPTurbojet class structure
        default_od = [
            {'name': 'OD0', 'alt': 0.0, 'mn': 0.000001, 'fn': 11000.0},
            {'name': 'OD1', 'alt': 5000.0, 'mn': 0.2, 'fn': 8000.0}
        ]
        
        for i, od_data in enumerate(default_od):
            row = i + 1
            ttk.Label(od_frame, text=od_data['name']).grid(row=row, column=0, padx=5, pady=2)
            
            alt_var = tk.DoubleVar(value=od_data['alt'])
            mn_var = tk.DoubleVar(value=od_data['mn'])
            fn_var = tk.DoubleVar(value=od_data['fn'])
            
            ttk.Entry(od_frame, textvariable=alt_var, width=10).grid(row=row, column=1, padx=2)
            ttk.Entry(od_frame, textvariable=mn_var, width=10).grid(row=row, column=2, padx=2)
            ttk.Entry(od_frame, textvariable=fn_var, width=10).grid(row=row, column=3, padx=2)
            
            self.od_vars.append({'name': od_data['name'], 'alt': alt_var, 'mn': mn_var, 'fn': fn_var})

        # --- CONTROLS ---
        btn_frame = ttk.Frame(input_panel)
        btn_frame.pack(pady=20, fill='x')
        
        run_btn = ttk.Button(btn_frame, text="RUN ANALYSIS", command=self.run_analysis)
        run_btn.pack(side='left', expand=True, fill='x', padx=5)
        
        # --- OUTPUT ---
        ttk.Label(output_panel, text="Simulation Output", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky="w")
        
        self.output_text = tk.Text(output_panel, font=("Consolas", 9), state='disabled', wrap='none')
        self.output_text.grid(row=1, column=0, sticky="nsew")
        
        # Scrollbars
        ys = ttk.Scrollbar(output_panel, orient='vertical', command=self.output_text.yview)
        ys.grid(row=1, column=1, sticky='ns')
        xs = ttk.Scrollbar(output_panel, orient='horizontal', command=self.output_text.xview)
        xs.grid(row=2, column=0, sticky='ew')
        self.output_text['yscrollcommand'] = ys.set
        self.output_text['xscrollcommand'] = xs.set

    def add_entry(self, parent, label, var):
        f = ttk.Frame(parent)
        f.pack(fill='x', pady=2)
        ttk.Label(f, text=label, width=20).pack(side='left')
        ttk.Entry(f, textvariable=var).pack(side='right', expand=True, fill='x')

    def log(self, msg):
        self.output_text.config(state='normal')
        self.output_text.insert('end', msg + "\n")
        self.output_text.see('end')
        self.output_text.config(state='disabled')
        self.root.update()

    def clear_log(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, 'end')
        self.output_text.config(state='disabled')

    def run_analysis(self):
        self.clear_log()
        self.log("Initializing Problem...")
        
        try:
            prob = om.Problem()
            mp_turbojet = prob.model = MPTurbojet()
            prob.setup(check=False)
            
            # --- Set Design Values ---
            dv = self.design_vars
            prob.set_val('DESIGN.fc.alt', dv['alt'].get(), units='ft')
            prob.set_val('DESIGN.fc.MN', dv['mn'].get())
            prob.set_val('DESIGN.balance.Fn_target', dv['fn'].get(), units='lbf')
            prob.set_val('DESIGN.balance.T4_target', dv['t4'].get(), units='degR')
            prob.set_val('DESIGN.comp.PR', dv['pr'].get())
            prob.set_val('DESIGN.comp.eff', dv['eff_comp'].get())
            prob.set_val('DESIGN.turb.eff', dv['eff_turb'].get())
            
            # Guesses (Hardcoded from original script to ensure stability)
            prob['DESIGN.balance.FAR'] = 0.01755
            prob['DESIGN.balance.W'] = 168.45
            prob['DESIGN.balance.turb_PR'] = 4.46
            
            # --- Set Off-Design Values ---
            for od in self.od_vars:
                pt_name = od['name']
                prob.set_val(f'{pt_name}.fc.alt', od['alt'].get(), units='ft')
                prob.set_val(f'{pt_name}.fc.MN', od['mn'].get())
                prob.set_val(f'{pt_name}.balance.Fn_target', od['fn'].get(), units='lbf')
                
                # Guesses for OD
                prob[f'{pt_name}.balance.W'] = 166.0
                prob[f'{pt_name}.balance.FAR'] = 0.0168
                prob[f'{pt_name}.balance.Nmech'] = 8197.0
                prob[f'{pt_name}.turb.PR'] = 4.6

            self.log("Running Solver...")
            
            # Suppress console output during run
            prob.set_solver_print(level=-1)
            prob.set_solver_print(level=2, depth=1)
            
            import time
            st = time.time()
            prob.run_model()
            dur = time.time() - st
            
            self.log(f"Done in {dur:.2f} seconds.")
            self.log("="*60)
            
            # --- Generate Reports ---
            points = ['DESIGN'] + mp_turbojet.od_pts
            for pt in points:
                report = get_viewer_output(prob, pt)
                self.log(report)
                self.log("\n")
                
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleTurbojetGUI(root)
    root.mainloop()