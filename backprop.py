# Back-propagation VMC Test Run
import datetime
import numpy as np

class QuantumSystem:
    # A system consists of nuclei and electrons.
    # Default parameters.
    NUMBER_OF_DETS = 5
    GAUSSIANS_PER_ORBITAL = 5
    VMC_STEPS = 5000
    EPS = 1.0e-3

    def __init__(self, numberOfDets = None, gaussianPerOrbital = None):
        self.nuclei = None
        self.electrons = None

        if numberOfDets is None:
            numberOfDets = QuantumSystem.NUMBER_OF_DETS
        if gaussianPerOrbital is None:
            gaussianPerOrbital = QuantumSystem.GAUSSIANS_PER_ORBITAL
        self.numberOfDets = numberOfDets
        self.gaussianPerOrbital = gaussianPerOrbital

    def setNuclei(self, nucleiPositions, nucleiCharges):
        numberOfNuclei = nucleiPositions.shape[0]
        self.nuclei = {
            'positions': nucleiPositions,
            'charges': nucleiCharges,
            'numberOfNuclei': numberOfNuclei
        }

        # Get nuclei-nuclei energy.
        energy = 0.0
        for i in range(numberOfNuclei):
            for j in range(i + 1, numberOfNuclei):
                r = nucleiPositions[j] - nucleiPositions[i]
                energy += nucleiCharges[i] * nucleiCharges[j] / np.linalg.norm(r)
        self.nuclei['energy'] = energy

    def setNumberOfElectrons(self, numberOfUpElectrons, numberOfDownElectrons):
        self.electrons = {
            'numberOfUpElectrons': numberOfUpElectrons,
            'numberOfDownElectrons': numberOfDownElectrons,
            'numberOfElectrons': numberOfUpElectrons + numberOfDownElectrons
        }

    def solve(self):
        # Obtain the wave function of the electrons.
        print 'Solving...'
        print datetime.datetime.now(), 'Start.'

        # Each gaussian is determined by 5 parameters: mu (3d-vector), sigma, and amplitude.
        parameters = np.random.rand(
            self.numberOfDets, self.electrons['numberOfElectrons'], self.gaussianPerOrbital, 5)

        # TODO: Optimize wave function.

        self.parameters = parameters
        print datetime.datetime.now(), 'Complete.'

    def _getEnergyAndVariance(self, steps):
        # Get the energy of electron hamiltonian (excluding nuclei-nuclei part).
        R = np.random.rand(self.electrons['numberOfElectrons'], 3) # x, y, z coordinates.
        localEnergies = np.zeros(steps)
        for i in range(steps):
            R = self._getNextR(R)
            Psai = self._getPsai(R)
            HPsai = self._getHPsai(R)
            localEnergy = HPsai / Psai
            localEnergies[i] = localEnergy
            if i % 100 == 0:
                print 'MEAN, STD:', np.mean(localEnergies), np.std(localEnergies)
        return np.mean(localEnergies), np.std(localEnergies)

    def _getNextR(self, R):
        newR = R + np.random.normal(0, 1.0, 3)
        acceptanceRatio = self._getPsai(newR) / self._getPsai(R)
        if acceptanceRatio > np.random.rand():
            return newR
        return R

    def _getElement(self, R, detId, positionId, orbitalId):
        position = R[positionId]
        orbital = self.parameters[detId][orbitalId]
        mus = orbital[:, 0:3]
        sigmas = orbital[:, 3]
        weights = orbital[:, 4]
        rs = np.sum((position - mus)**2, axis = 1)
        gaussians = np.exp(-rs / (2.0 * sigmas**2))
        return np.sum(gaussians * weights)

    def _getPsai(self, R):
        dets = 0.0
        for i in range(self.numberOfDets):
            numberOfElectrons = self.electrons['numberOfElectrons']
            det = np.zeros((numberOfElectrons, numberOfElectrons))
            for j in range(numberOfElectrons):
                for k in range(numberOfElectrons):
                    det[j][k] = self._getElement(R, i, j, k)
            dets += np.linalg.det(det)
        return dets

    def _getD2Psai(self, R):
        D2Psai = 0.0
        eps = QuantumSystem.EPS
        eps2 = eps**2
        dRs = np.eye(3) * eps
        for i in range(self.electrons['numberOfElectrons']):
            for j in range(3): # x, y, z.
                Psais = np.zeros(5)
                for k in range(5): # Five-point stencils approximation.
                    Psais[k] = self._getPsai(R + dRs[j] * (k - 2))
                product = Psais * np.array([-1, 16, -30, 16, -1])
                D2Psai += np.sum(product) / (12 * eps2)
        return D2Psai

    def _getHPsai(self, R):
        HPsai = 0.0
        Psai = self._getPsai(R)
        numberOfElectrons = self.electrons['numberOfElectrons']
        numberOfNuclei = self.nuclei['numberOfNuclei']

        # Get electron-electron energy.
        H_ee = 0.0
        for i in range(numberOfElectrons):
            for j in range(i + 1, numberOfElectrons):
                H_ee += 1 / np.linalg.norm(R[i] - R[j])
        HPsai += H_ee * Psai

        # Get electron-nuclei energy.
        H_en = 0.0
        for i in range(numberOfNuclei):
            for j in range(numberOfElectrons):
                nucleusPosition = self.nuclei['positions'][i]
                r = nucleusPosition[i] - R[j]
                H_en += -self.nuclei['charges'][i] / np.linalg.norm(r)
        HPsai += H_en * Psai

        # Get the dynamic energy.
        HPsai += -0.5 * self._getD2Psai(R)

        return HPsai

    def getEnergy(self):
        print 'Evaluating energy...'
        energy, variance = self._getEnergyAndVariance(QuantumSystem.VMC_STEPS)
        energy += self.nuclei['energy']
        return energy, variance # In a.u.

def main():
    print 'Back-propagation VMC', datetime.datetime.now()

    # First line for name of the system.
    systemName = raw_input()
    print 'System name:', systemName

    if systemName == 'test':
        numberOfNuclei = 2
        nuclei = np.array([[0.0, 0.0, 0.0], [1.4632, 0.0, 0.0]])
        nucleiCharges = np.array([2, 1])
        numberOfUpElectrons, numberOfDownElectrons = (1, 1)
    else:
        # Next line for number of nuclei.
        numberOfNuclei = int(raw_input())
        nuclei = np.zeros((numberOfNuclei, 3))

        # Next line for charge of each nucleus, separated by spaces.
        nucleiCharges = np.asarray(map(int, raw_input().split()))

        # Nuclei coordinates.
        for i in range(numberOfNuclei):
            nuclei[i] = map(float, raw_input().split()) # One nucleus per line. In a.u.

        # Next line for number of up and down electrons.
        numberOfUpElectrons, numberOfDownElectrons = map(int, raw_input().split())


    print 'Nuclei charges:', nucleiCharges
    print 'Nuclei coordinates (a.u.):'
    print nuclei
    print 'Number of electrons (up, down):', numberOfUpElectrons, numberOfDownElectrons

    quantumSystem = QuantumSystem()
    quantumSystem.setNuclei(nuclei, nucleiCharges)
    quantumSystem.setNumberOfElectrons(numberOfUpElectrons, numberOfDownElectrons)
    quantumSystem.solve()

    energy, variance = quantumSystem.getEnergy()
    print 'Energy (a.u.):', energy, '+-', variance

if __name__ == '__main__':
    main()