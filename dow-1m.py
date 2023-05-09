#!/home/fmgarmor/miot_env/bin/python3

import subprocess
import time
import urllib.request

def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,shell=True) #in py 3.5: universal_newlines=True; in 3.7+ text=True
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

i=0
while True:
    print("Ejecucion:", i+1)

    #GitHub
    #runcmd("wget -P downloads/LIBLEMQHTTP/ https://github.com/frangam/tizensor/releases/download/1.0/LIBLEMQHTTP.a", verbose=False)
    #runcmd("wget -P downloads/Tizensor-SmartwatchAPP/ https://github.com/frangam/tizensor/releases/download/1.0/SmartwatchAPP.tpk", verbose=False)
    #runcmd("wget -P downloads/iOSAPP/ https://github.com/frangam/tizensor/releases/download/1.0/iOSAPP.ipa", verbose=False)
    runcmd("wget -P downloads/jnlp/ https://github.com/frangam/JNLP/releases/download/1.0.0/JNLP-1.0.1.jar")
    runcmd("wget -P downloads/towerdefense/ https://github.com/frangam/TowerDefense/releases/download/1.0/TowerDefenseTest_Web.zip", verbose=False)
    #runcmd("wget -P downloads/guimultigithub/ https://github.com/frangam/gui-multiresolucion-unity/releases/download/1.0.0/gui-multiresolucion-unity-v1.0.0.zip", verbose=False)
    #runcmd("wget -P downloads/wear-sensor-ml/ https://github.com/frangam/wearable-sensor-ml-pipeline/releases/download/1.0/wearable-sensor-ml-pipeline-1.0.zip", verbose=False)
    #runcmd("wget -P downloads/cutaneous-melanoma/ https://github.com/frangam/genomic-classification-cutaneous-melanoma/releases/download/1.0/genomic-classification-cutaneous-melanoma-v1.0.zip", verbose=False)
    #runcmd("wget -P downloads/unity-vox/ https://github.com/frangam/unity-voxel-engine/releases/download/1.0/VoxelEngine1.0.unitypackage", verbose=False)
    runcmd("wget -P downloads/ACOForVRP/ https://github.com/frangam/ACOForVRP/releases/download/1.0/ACOForVRP-v1.0.zip", verbose=False)
    #runcmd("wget -P downloads/unity-gametemplate/ https://github.com/frangam/unity-game-template/releases/download/2.2.0/GameTemplate_v2.2.0.unitypackage.gz", verbose=False)
    
    #takes more time
    #runcmd("wget -P garmo_downloads_py/allerdet-github/ https://github.com/frangam/ALLERDET/releases/download/1.0/ALLERDET-1.0.zip", verbose=False)

   
    time.sleep(30) # Sleep for X seconds
    runcmd("rm -r downloads/", verbose=False)

    time.sleep(30) # Sleep for X seconds
    i+=1