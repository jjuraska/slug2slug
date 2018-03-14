import json
import re
from nltk.stem.wordnet import WordNetLemmatizer


def extract_subcategories(filename, category):
    subcategories = {}

    stemmer = WordNetLemmatizer()

    with open(filename, 'r') as f_categories:
        data = json.load(f_categories)

        for cat in data:
            if category in cat['parents']:
                # split category title with '/' into multiple separate titles
                if '/' in cat['title']:
                    cat_titles = [cat_separated.strip() for cat_separated in cat['title'].split('/')]
                else:
                    cat_titles = [cat['title']]

                for cat_title in cat_titles:
                    # remove information in parentheses from the category title
                    cat_title = re.sub(r'\s*\(.*\)', '', cat_title.lower())

                    # transform plurals into singulars
                    cat_title_stem = ' '.join([stemmer.lemmatize(w) for w in cat_title.split()])

                    # save subcategory along with its ID
                    if cat_title not in subcategories:
                        subcategories[cat_title] = []
                    subcategories[cat_title].append(cat['alias'])

                    if cat_title_stem != cat_title:
                        if cat_title_stem not in subcategories:
                            subcategories[cat_title_stem] = []
                        subcategories[cat_title_stem].append(cat['alias'])

    with open('data/yelp/categories_' + category + '.json', 'w') as f_dump:
        json.dump(subcategories, f_dump, indent=4)


if __name__ == '__main__':
    filename = 'data/yelp/categories.json'

    extract_subcategories(filename, 'restaurants')
