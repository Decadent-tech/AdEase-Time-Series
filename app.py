import main as m

print(len(m.train), len(m.ex[:520]))  # Should be equal
print(len(m.test), len(m.ex[520:]))   # Should be equal
