import subprocess

def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,shell=True) #in py 3.5: universal_newlines=True; in 3.7+ text=True
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass
