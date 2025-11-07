from qiskit import QuantumCircuit

class MaryGate:
    def __init__(self, circ, angle, target, controls, bin,gate_type,ancilla = None):
        self.circ = circ
        self.angle = angle
        self.target = target
        self.controls = controls
        self.ancilla = ancilla
        self.gate_type = gate_type
        self.bin = bin

    def apply_gate(self):
        if self.gate_type == 'mary5':
            self.mary5_gate()
        elif self.gate_type == 'mary7':
            self.mary7_gate()
        elif self.gate_type == 'mary8':
            self.mary_8_gate()
        elif self.gate_type == 'mary9':
            self.mary9_gate()
        elif self.gate_type == 'mary10':
            self.mary10_gate()
        elif self.gate_type == 'mary32':
            self.mary32_gate()
        elif self.gate_type == 'mary128':
            self.mary128_gate()
        elif self.gate_type == 'mary256':
            self.mary256_gate()
        else:
            raise ValueError(f"Unsupported gate_type: {self.gate_type}")

    #images 4x4
    def mary5_gate(self):
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.cx(self.controls[0], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)
        self.circ.rccx(self.controls[1], self.controls[2], self.target)
        self.circ.rz(self.angle / 4, self.target)
        self.circ.cx(self.controls[3], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.rccx(self.controls[1], self.controls[2], self.target)
        self.circ.rz(self.angle / 4, self.target)
        self.circ.cx(self.controls[3], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.cx(self.controls[0], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)

    #images 8x8
    def mary7_gate(self):
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rccx(self.controls[0], self.controls[1], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)
        self.circ.rccx(self.controls[2], self.controls[3], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rccx(self.controls[4], self.controls[5], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.rccx(self.controls[2], self.controls[3], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rccx(self.controls[4], self.controls[5], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rccx(self.controls[0], self.controls[1], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)

    def mary_8_gate(self):
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rccx(self.controls[0], self.controls[1], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)
        self.circ.rccx(self.controls[2], self.controls[3], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rcccx(self.controls[4], self.controls[5], self.controls[6], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.rccx(self.controls[2], self.controls[3], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rcccx(self.controls[4], self.controls[5], self.controls[6], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rccx(self.controls[0], self.controls[1], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)

    #images 16x16
    def mary9_gate(self):
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rccx(self.controls[0], self.controls[1], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)
        self.circ.rcccx(self.controls[2], self.controls[3], self.controls[4], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rcccx(self.controls[5], self.controls[6], self.controls[7], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.rcccx(self.controls[2], self.controls[3], self.controls[4], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rcccx(self.controls[5], self.controls[6], self.controls[7], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rccx(self.controls[0], self.controls[1], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)

    def mary10_gate(self):
        if self.ancilla is not None and self.controls[0] == self.ancilla:
            ancilla_qubit = self.controls[0]
        else:
            ancilla_qubit = None


        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rcccx(self.controls[0], self.controls[1], self.controls[2], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)
        self.circ.rcccx(self.controls[3], self.controls[4], self.controls[5], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rcccx(self.controls[6], self.controls[7], self.controls[8], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.rcccx(self.controls[3], self.controls[4], self.controls[5], self.target)

        self.circ.rz(self.angle / 4, self.target)
        self.circ.rcccx(self.controls[6], self.controls[7], self.controls[8], self.target)

        self.circ.rz(-self.angle / 4, self.target)
        self.circ.h(self.target)
        self.circ.t(self.target)
        self.circ.rcccx(self.controls[0], self.controls[1], self.controls[2], self.target)

        self.circ.tdg(self.target)
        self.circ.h(self.target)
    
    #images 32x32
    def mary32_gate(self):
        # Applying H gate to the control qubits
        #self.circ.h(self.controls)
        clist = []

        for i in self.bin:
            clist.append(int(i))

        for i in range(len(clist)):
            if clist[i] == 0:
                self.circ.x(self.controls[-i-1])

        # Applying the first rccx operation
        self.circ.rccx(self.controls[4], self.controls[5], self.ancilla)
        self.circ.x(self.controls[4])
        self.circ.x(self.controls[5])

        # Applying the second rccx operation
        self.circ.rccx(self.controls[6], self.controls[7], self.controls[4])
        self.circ.rccx(self.controls[8], self.controls[9], self.controls[5])

        # Creating new controls for the Mary8_gate
        new_controls = [self.ancilla] + self.controls[:6]

        mary_gate = MaryGate(circ=self.circ, angle=self.angle, target=self.target, controls=new_controls,bin= None, gate_type='mary8',ancilla=self.ancilla)
        mary_gate.apply_gate()

        # Reversing the operations
        self.circ.rccx(self.controls[8], self.controls[9], self.controls[5])
        self.circ.rccx(self.controls[6], self.controls[7], self.controls[4])
        self.circ.x(self.controls[5])
        self.circ.x(self.controls[4])

        # Reversing the first rccx operation
        self.circ.rccx(self.controls[4], self.controls[5], self.ancilla)

        for i in range(len(clist)):
            if clist[i] == 0:
                self.circ.x(self.controls[-i-1])

    #image 128x128
    def mary128_gate(self):
        # Applying H gate to the control qubits
        self.circ.h(self.controls)

        # Applying the first rccx operation
        self.circ.rccx(self.controls[4], self.controls[5], self.ancilla)
        self.circ.x(self.controls[4])
        self.circ.x(self.controls[5])

        # Applying the second rccx operation
        self.circ.rcccx(self.controls[6], self.controls[7], self.controls[8],self.controls[4])
        self.circ.rcccx(self.controls[9], self.controls[10],self.controls[11], self.controls[5])

        self.circ.x(self.controls[6])
        self.circ.rccx(self.controls[12], self.controls[13], self.controls[6])
                    
        # Creating new controls for the Mary8_gate
        new_controls = [self.ancilla] + self.controls[:7]

        mary_gate = MaryGate(circ=self.circ, angle=self.angle, target=self.target, controls=new_controls,bin=None,gate_type='mary9',ancilla=self.ancilla)
        mary_gate.apply_gate()

        # Reversing the operations
        self.circ.rccx(self.controls[12], self.controls[13], self.controls[6])
        self.circ.rcccx(self.controls[9], self.controls[10],self.controls[11], self.controls[5])
        self.circ.x(self.controls[5])
        self.circ.x(self.controls[4])
        self.circ.rcccx(self.controls[6], self.controls[7], self.controls[8],self.controls[4])
        self.circ.x(self.controls[4])


        # Reversing the first rccx operation
        self.circ.rccx(self.controls[4], self.controls[5], self.ancilla)


    #image 256x256
    def mary256_gate(self):
        # Applying H gate to the control qubits
        self.circ.h(self.controls)

        # Applying the first rccx operation
        self.circ.rccx(self.controls[4], self.controls[5], self.ancilla)
        self.circ.x(self.controls[4])
        self.circ.x(self.controls[5])

        # Applying the second rccx operation
        self.circ.rcccx(self.controls[6], self.controls[7], self.controls[8],self.controls[4])
        self.circ.rcccx(self.controls[9], self.controls[10],self.controls[11], self.controls[5])

        self.circ.x(self.controls[6])
        self.circ.x(self.controls[7])
        self.circ.rccx(self.controls[12], self.controls[13], self.controls[6])
        self.circ.rccx(self.controls[14], self.controls[15], self.controls[7])
                    
        new_controls = [self.ancilla] + [2,3,4,5,6,7,8,9,10]

        mary_gate = MaryGate(circ=self.circ, angle=self.angle, target=self.target, controls=new_controls,bin=None,gate_type='mary10',ancilla = self.ancilla)
        mary_gate.apply_gate()

        # Reversing the operations
        self.circ.rccx(self.controls[14], self.controls[15], self.controls[7])
        self.circ.rccx(self.controls[12], self.controls[13], self.controls[6])
        self.circ.rcccx(self.controls[9], self.controls[10],self.controls[11], self.controls[5])
        self.circ.x(self.controls[7])
        self.circ.x(self.controls[6])
        self.circ.x(self.controls[5])
        
        self.circ.rcccx(self.controls[6], self.controls[7], self.controls[8],self.controls[4])
        self.circ.x(self.controls[4])


        # Reversing the first rccx operation
        self.circ.rccx(self.controls[4], self.controls[5], self.ancilla)

