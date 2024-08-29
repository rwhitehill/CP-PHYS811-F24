def get_roots(a,b,c):
    term1 = -b/2/a
    D = b**2 - 4*a*c
    if D > 0:
        term2 = D**0.5/2/a
        print(f'x_{{+}} = {term1+term2}, x_{{-}} = {term1-term2}')
    elif D == 0:
        print(f'x_{{+}} = x_{{-}} = {term1}')
    else:
        term2 = (-D)**0.5/2/a
        print(f'x_{{+}} = {complex(term1,term2)}, x_{{-}} = {complex(term1,-term2)}')

    return None


if __name__ == '__main__':

    c = input('Enter the (real) coefficients for a quadratic polynomial: ')
    c = [int(_) for _ in c.split()]
    get_roots(*c)


