
# posisjoner er i km
# hastigheter er i km/s
solar_mass = 1988500e24


data = {
    "sun": {
        "mass": 1988500e24,
        "x": -1.307670426962003E+06,
        "y": -2.001620916338117E+05,
        "z": 3.210857821818738E+04,
        "vx": 4.750221642521828E-03,
        "vy": -1.447494113088643E-02,
        "vz": 1.221075509220525E-05,
    },
    "mercury": {
        "mass": 3.302e23,
        "x": 5.084724420867205E+07,
        "y": 3.289648871774699E+06,
        "z": -4.466591153301381E+06,
        "vx": -1.265614216002625E+01,
        "vy": 5.076801513305359E+01,
        "vz": 5.311298706265198E+00,
    },
    "venus": {
        "mass": 48.685e23,
        "x": -7.668375488514954E+07,
        "y": -7.783881542748766E+07,
        "z": 3.315563254841864E+06,
        "vx": 2.488472691304460E+01,
        "vy": -2.457353170424098E+01,
        "vz": -1.772812788341042E+00,
    },
    "earth": {
        "mass": 5.97219e24,
        "x": -1.889479143726501E+07,
        "y": -1.511260385729528E+08,
        "z": 4.002018439576030E+04,
        "vx": 2.911765262001637E+01,
        "vy": -3.584413785108487E+00,
        "vz": -1.039553509460056E-03,
    },
    "mars": {
        "mass": 6.4171e23,
        "x": -2.390435740523276E+08,
        "y": 7.364631953540286E+07,
        "z": 7.411359172833487E+06,
        "vx": -6.276389448814106E+00,
        "vy": -2.108544169040553E+01,
        "vz": -2.875189518284200E-01,
    },
    "jupiter": {
        "mass": 1.898e27,
        "x": 6.574878292951905E+08,
        "y": 3.399020006403722E+08,
        "z": -1.611999093734986E+07,
        "vx": -6.147679413283897E+00,
        "vy": 1.222261125986798E+01,
        "vz": 8.679438420783381E-02,
    },
    "saturn": {
        "mass": 5.6834e26,
        "x": 1.281643441442537E+09,
        "y": -7.063971104050153E+08,
        "z": -3.874585005793929E+07,
        "vx": 4.122033853428058E+00,
        "vy": 8.442225737100351E+00,
        "vz": -3.114508310860078E-01,
    },
    "uranus": {
        "mass": 86.813e24,
        "x": 1.925739084664315E+09,
        "y": 2.218557799464732E+09,
        "z": -1.670852236202729E+07,
        "vx": -5.192773333684966E+00,
        "vy": 4.146757231082335E+00,
        "vz": 8.270065736680365E-02,
    },
    "neptune": {
        "mass": 102.409e24,
        "x": 4.457249477309738E+09,
        "y": -3.625933466499828E+08,
        "z": -9.525522218242368E+07,
        "vx": 4.047526058274553E-01,
        "vy": 5.449627323513529E+00,
        "vz": -1.215498535244772E-01,
    },
    "pluto": {
        "mass": 1.307e22,
        "x": 2.488892694468043E+09,
        "y": -4.564116187934589E+09,
        "z": -2.315468476838567E+08,
        "vx": 4.923259570764408E+00,
        "vy": 1.416689316297159E+00,
        "vz": -1.591776283816501E+00,
    },
}

km_per_au = 1.496e+8
seconds_per_year = 365.25 * 24 * 60 * 60

for planet in data:
    data[planet]["x"] /= km_per_au
    data[planet]["y"] /= km_per_au
    data[planet]["z"] /= km_per_au

    data[planet]["vx"] /= (km_per_au * seconds_per_year)
    data[planet]["vy"] /= (km_per_au * seconds_per_year)
    data[planet]["vz"] /= (km_per_au * seconds_per_year)

    data[planet]["mass"] /= solar_mass

print(data)

M = sum([data.get(planet).get("mass") for planet in data])
print(M)


fname = "solsystem_masser.txt"
with open(fname, "w") as outfile:
    outfile.write("navn masse\n")
    for planet in data:
        outfile.write(f"{planet} {data.get(planet).get('mass')}")
        outfile.write("\n")


fname = "solsystem_posisjoner.txt"
with open(fname, "w") as outfile:
    outfile.write("navn x y z")
    for planet in data:
        x = data.get(planet).get("x")
        y = data.get(planet).get("y")
        z = data.get(planet).get("z")
        outfile.write(f"{planet} {x} {y} {z}")
        outfile.write("\n")

fname = "solsystem_hastigheter.txt"
with open(fname, "w") as outfile:
    outfile.write("navn vx vy vz")
    for planet in data:
        vx = data.get(planet).get("vx")
        vy = data.get(planet).get("vy")
        vz = data.get(planet).get("vz")
        outfile.write(f"{planet} {vx} {vy} {vz}")

        


 