from ignite.metrics import Rouge

if __name__ == "__main__":
    m = Rouge(variants=["L"], multiref="average")

    candidate = "the cat is not there".split()
    references = [
        "the cat is on the mat".split(),
        "there is a cat on the mat".split()
    ]

    m.update(([candidate], [references]))

    print(m.compute())