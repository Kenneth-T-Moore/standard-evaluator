import openmdao.api as om
from openmdao.test_suite.scripts.circuit_analysis import Resistor, Diode, Node


class Circuit(om.Group):
    def setup(self):
        self.add_subsystem(
            "n1", Node(n_in=1, n_out=2), promotes_inputs=[("I_in:0", "I_in")]
        )
        self.add_subsystem("n2", Node())  # leaving defaults

        self.add_subsystem("R1", Resistor(R=100.0), promotes_inputs=[("V_out", "Vg")])
        self.add_subsystem("R2", Resistor(R=10000.0))
        self.add_subsystem("D1", Diode(), promotes_inputs=[("V_out", "Vg")])

        self.connect("n1.V", ["R1.V_in", "R2.V_in"])
        self.connect("R1.I", "n1.I_out:0")
        self.connect("R2.I", "n1.I_out:1")

        self.connect("n2.V", ["R2.V_out", "D1.V_in"])
        self.connect("R2.I", "n2.I_in:0")
        self.connect("D1.I", "n2.I_out:0")

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 20
        self.linear_solver = om.DirectSolver()

    def create_model(self):
        # Setup problem and model
        self.p = om.Problem()
        self.model = self.p.model
        #
        self.model.set_input_defaults("ground.V", 0.0, units="V")
        self.model.set_input_defaults("source.I", 0.1, units="A")
        self.model.add_subsystem(
            "circuit",
            Circuit(),
            promotes_inputs=[("Vg", "ground.V"), ("I_in", "source.I")],
        )
        #
        self.model.add_design_var("ground.V")
        self.model.add_design_var("source.I")
        self.model.add_objective("circuit.D1.I")

        self.p.setup()
        self.p.check_config(checks=["unconnected_inputs"], out_file=None)
        #
        # set some initial guesses
        self.p["circuit.n1.V"] = 10.0
        self.p["circuit.n2.V"] = 1.0
