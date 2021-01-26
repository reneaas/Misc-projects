import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

files = ["results/one_body_density_interacting_1.txt",\
        "results/one_body_density_interacting_10.txt"]
        # "results/one_body_density_interacting_100.txt",\
        # "results/one_body_density_interacting_50.txt",\


R = []

for infilename in files:
    r = []
    with open(infilename, "r") as infile:
        lines = infile.readlines()

        for line in lines:
            vals = line.split()
            r.append(float(vals[0]))

    R.append(r)


# R = np.array(R)
R = np.array(R)
R = R.T
df = pd.DataFrame(R, columns = ["non-interacting; n = 1", "interacting; n = 10"])
# df = pd.DataFrame(R, index=["non-interacting", "interacting"])
# print(df)
fontsize = 14
# sns.displot(df, kind="kde", shade=True)
sns.kdeplot(x=df["non-interacting; n = 1"], shade=True, label="non-interacting")
# sns.kdeplot(x=df["interacting; n = 10"], shade=True, label="interacting; n = 10")
# sns.kdeplot(x=df["interacting; n = 50"], shade=True, label="interacting; n = 50")
sns.kdeplot(x=df["interacting; n = 100"], shade=True, label="interacting; n = 100")
plt.xlabel(r"$r$", fontsize=fontsize)
plt.ylabel(r"$\rho(r)$", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.show()
