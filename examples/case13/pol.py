import os
import numpy as np
import xml.etree.ElementTree as ET

def main():
    pvd_file = "sol/sol.pvd"
    root = ET.parse(pvd_file).getroot()[0]

    pty = "$HOME/ParaView-5.8.1-MPI-Linux-Python3.7-64bit/bin/pvpython "
    script = "plot_pv.py "

    times = []
    for r in root:
        vtu_file = r.attrib["file"]
        times.append(r.attrib["timestep"])
        out_file = "./pol/" + vtu_file[:-3] + "csv"

        os.system(pty + script + vtu_file + " " + out_file)

    np.savetxt("./pol/times.csv")

if __name__ == "__main__":
    main()
