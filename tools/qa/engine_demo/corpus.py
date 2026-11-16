"""The documents the demo ingests and the word stems a right answer about each one uses."""

CORPUS = {
    "bees.md": (
        "# Honeybees\n\nA forager bee that finds flowers returns to the hive and performs a "
        "waggle dance. The angle of the dance shows the direction of the flowers and its "
        "length shows the distance.\n"
    ),
    "tides.md": (
        "# Tides\n\nThe moon pulls on the ocean and makes two high tides and two low tides "
        "in most places each day.\n"
    ),
    "bread.md": (
        "# Bread\n\nSourdough rises because wild yeast and bacteria in the starter make gas "
        "and acid while the dough rests.\n"
    ),
}

# Lower-case word stems, per file, that an answer drawn from that file uses. A word matches a
# stem when it starts with it, so "dancing" and "waggling" match without a list of forms.
CORPUS_FACTS = {
    "bees.md": ("waggl", "danc", "forager", "perform", "hive"),
    "tides.md": ("tide", "moon", "pull"),
    "bread.md": ("yeast", "sourdough", "starter", "bacteri", "dough"),
}
