def innlesning():
    handlevogn = []
    while True:
        vare = input('Hva vil du kjøpe?\n')
        if vare == "":
            break
        try:
            antall = int(input('Hvor mange enheter ønsker du å handle?\n'))
            pris = float(input('Hva koster varen?\n'))
            kostnad = (antall * pris)
            handlevogn.append((vare, antall, pris))
            print(handlevogn)
        except ValueError:
            print("Feil")
        finally:
            print(kostnad)
    return handlevogn
a = innlesning()
def utskrift(vareliste):
    print('Beskrivelse', 'Linjekostnad', sep='\t\t\t\t')
    print('-------------------------------------')
    total = 0
    for x in vareliste:
        pris = x[1] * x[2]
        print(x[0], pris, sep='\t\t\t\t\t')
        total += pris
    print('-------------------------------------')
    print('Total', total, sep='\t\t\t\t\t')
utskrift(a)