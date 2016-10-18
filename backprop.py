# Back-propagation VMC Test Run
import datetime
import numpy as np

class QuantumSystem:
    # A system consists of nuclei and electrons.
    # Default parameters.
    NUMBER_OF_DETS = 5
    GAUSSIANS_PER_ORBITAL = 5
    VMC_STEPS = 1000

    def __init__(self):
        self.nuclei = None
        self.nucleiCharges = None
        self.numberOfElectrons = 0
        self.numberOfUpElectrons = 0
        self.numberOfDownElectrons = 0
        self.nnEnergy = 0.0

        # Default configs.
        self.numberOfDets = QuantumSystem.NUMBER_OF_DETS
        self.gaussianPerOrbital = QuantumSystem.GAUSSIANS_PER_ORBITAL

    def setNuclei(self, nuclei, nucleiCharges):
        self.nuclei = nuclei
        self.nucleiCharges = nucleiCharges
        numberOfNuclei = nuclei.shape[0]
        self.numberOfNuclei = numberOfNuclei

        # Get nuclei-nuclei energy.
        nnEnergy = 0.0
        for i in range(numberOfNuclei):
            for j in range(i + 1, numberOfNuclei):
                r = nuclei[j] - nuclei[i]
                nnEnergy += nucleiCharges[i] * nucleiCharges[j] / np.linalg.norm(r)
        self.nnEnergy = nnEnergy

    def setNumberOfElectrons(self, numberOfUpElectrons, numberOfDownElectrons):
        self.numberOfUpElectrons = numberOfUpElectrons
        self.numberOfDownElectrons = numberOfDownElectrons
        self.numberOfElectrons = numberOfUpElectrons + numberOfDownElectrons

    def solve(self):
        # Obtain the wave function of the electrons.
        print 'Solving...'
        print datetime.datetime.now(), 'Start.'

        # Each gaussian is determined by three parameters: mu, sigma, and amplitude
        parameters = np.random.rand(
            self.numberOfDets, self.numberOfElectrons, self.gaussianPerOrbital, 3)

        # TODO: Optimize wave function.

        self.parameters = parameters
        print datetime.datetime.now(), 'Complete.'

    def _getEnergyAndVariance(self, steps):
        # Get the energy of electron hamiltonian (excluding nuclei-nuclei part).
        R = np.random.rand(self.numberOfElectrons, 3) # x, y, z coordinates.
        localEnergies = np.zeros(steps)
        for i in range(steps):
            R = self._getNextR(R)
            Psai = self._getPsai(R)
            HPsai = self._getHPsai(R)
            localEnergies[i] = HPsai / Psai
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
        mus = orbital[:, 0]
        sigmas = orbital[:, 1]
        weights = orbital[:, 2]
        gaussians = np.exp((position - mus) / (2.0 * sigmas * sigmas))
        return np.sum(gaussians * weights)

    def _getPsai(self, R):
        dets = 0.0
        for i in range(self.numberOfDets):
            numberOfElectrons = self.numberOfElectrons
            det = np.zeros(numberOfElectrons, numberOfElectrons)
            for j in range(numberOfElectrons):
                for k in range(numberOfElectrons):
                    det[j][k] = self._getElement(R, i, j, k)
            dets += np.linalg.det(det)
        return dets

    def _getD2Psai(self, R):
        
        print 'TODO'

    def _getHPsai(self, R):
        HPsai = 0.0
        Psai = self._getPsai(R)
        numberOfElectrons = self.numberOfElectrons
        numberOfNuclei = self.numberOfNuclei

        # Get electron-electron energy.
        Hee = 0.0
        for i in range(numberOfElectrons):
            for j in range(i + 1, numberOfElectrons):
                Hee += 1 / np.linalg.norm(R[i] - R[j])
        HPsai += Hee * Psai

        # Get electron-nuclei energy.
        Hen = 0.0
        numberOfElectrons = self.numberOfElectrons
        nuclei = self.nuclei
        nucleiCharges = self.nucleiCharges
        for i in range(numberOfNuclei):
            for j in range(numberOfElectrons):
                nucleusPosition = nuclei[i]
                Hen -=  nucleiCharges[i] / np.linalg.norm(nucleusPosition[i] - R[j])
        HPsai += Hen * Psai

        # Get the dynamic energy.
        HPsai += 0.5 * self._getD2Psai(R)

        return HPsai

    def getEnergy(self):
        energy, variance = self._getEnergyAndVariance(QuantumSystem.VMC_STEPS)
        energy += self.nnEnergy
        return energy, variance # In a.u.

def main():
    print 'Back-propagation VMC', datetime.datetime.now()

    # First line for name of the system.
    systemName = raw_input()
    print 'System name:', systemName

    # Next line for number of nuclei.
    numberOfNuclei = int(raw_input())
    nuclei = np.zeros((numberOfNuclei, 3))

    # Next line for charge of each nucleus, separated by spaces.
    nucleiCharges = np.asarray(map(int, raw_input().split()))
    print 'Nuclei charges:', nucleiCharges

    for i in range(numberOfNuclei):
        nuclei[i] = map(float, raw_input().split()) # One nucleus per line. In a.u.

    print 'Nuclei coordinates (a.u.):'
    print nuclei

    # Next line for number of up and down electrons.
    numberOfUpElectrons, numberOfDownElectrons = map(int, raw_input().split())
    print 'Number of electrons (up, down):', numberOfUpElectrons, numberOfDownElectrons

    quantumSystem = QuantumSystem()
    quantumSystem.setNuclei(nuclei, nucleiCharges)
    quantumSystem.setNumberOfElectrons(numberOfUpElectrons, numberOfDownElectrons)
    quantumSystem.solve()

    energy, variance = quantumSystem.getEnergy()
    print 'Energy (a.u.):', energy, '+-', variance

if __name__ == '__main__':
    main()