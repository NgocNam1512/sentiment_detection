def preprocess():
    test_sentences = []
    with open("raw_data.txt", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                text = line.strip()
            else:
                label = line.strip()
                score = label.split("\t")[1]
                sentence = f"{score},{text}"
                test_sentences.append(sentence)

    with open("train_data.txt", "w", encoding="utf8") as f:
        content = "\n".join(test_sentences)
        f.write(content + "\n")


if __name__ == "__main__":
    preprocess()