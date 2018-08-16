bla = [[0, 1, 3], [4, 2, 1]]
def norm(data):
    for row in range(len(data)):
        for col in range(len(data[row])):
            data[row][col] = data[row][col]/4
    return data

print(norm(bla))