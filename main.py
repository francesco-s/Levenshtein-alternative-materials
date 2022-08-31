import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from textdistance import levenshtein


def descriptive_analysis(df):
    print("Dataframe info ===============")
    print(df.info())
    ''' 1° critical issues: 
    df.info() shows 31 categorical features and only 1 float64 feature.
    Most features contain a numeric value followed by its unit of measure.
    Pandas will rightly read these values as categorical variables.
    The presence of the unit of measure causes categorical features to exist even when not needed.
        
    For example the categorical value "125V" of "Maximum AC Voltage Rating" variable 
    can be considered without the unit of measurement V. 
    Solution:
        # Remove all character from a to z, from A to Z
        df[categorical_column_toprocess] = df[categorical_column_toprocess].str.replace(r"[a-zA-Z]",'') 
    '''

    # working with NaN
    print("Missing values for each feature ===============")
    print(df.isnull().sum())
    print()
    print("Missing values for each feature (mean):", df.isnull().sum().mean())

    missing_values = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]

    print("Total missing values count:", missing_values)
    print("Total cells:", total_cells)
    print("Percentage of missing values: ", missing_values / total_cells)  # about 37% of missing values

    ''' 2° critical issues: 
    about 37% of missing values on our dataset.
    One solution might be to eliminate those variables with a percentage of missing values above a certain threshold.
    With a threshold of 50 % we will drop the features: "Additional Feature", "Body Breadth (mm) ",
    "Maximum DC Voltage Rating ", "Maximum Power Dissipation", "Pre-arcing time-Min (ms)", "Product Diameter", 
    "Rated Voltage (V) ", "Rated Voltage(DC) (V) ". 
    Obviously it's necessary to carry out the tuning of the threshold value, so as not to eliminate significant features.
    
    The remaining null fields can be filled using the mean, the median or the most frequent value in the column.
    '''

    # checking outliers (commented for better console visibility)
    # for column in df.columns:
    # print(df[column].value_counts())


def material_similarity(df):
    res = dict()
    for i in range(0, len(df)):
        # dict containing 5 material description (keys) with levenshtein distance as a value
        dis = {'null': 999, 'null1': 999, 'null2': 999, 'null3': 999, 'null4': 999}
        mat1 = df.iloc[i]['PART_DESCRIPTION']
        for j in range(0, len(df)):
            mat2 = df.iloc[j]['PART_DESCRIPTION']
            if mat2 != mat1:
                if not pd.isnull(mat1) and not pd.isnull(mat2):
                    lv = levenshtein(mat1, mat2)
                    max_value = dis[max(dis, key=dis.get)]
                    if max_value > lv:
                        dis[mat2] = lv
                        dis.pop(max(dis, key=dis.get))
        res[mat1] = dis # create a new entry on dict
        print(mat1, res[mat1])  # printing actual material and 5 similar materials

    # Output like this
    '''
    Fuse Miniature Fast Acting 1.6A 250V Holder Cartridge 5 X 20mm Ceramic Box CCC/PSE/VDE/cULus Electric Fuse, 
    Very Fast Blow, 1.6A, 250VAC, 1500A (IR), Inline/holder, 5x20mm
    
    {'Fuse Miniature Fast Acting 8A 250V Holder Cartridge 5 X 20mm Ceramic Box KC/PSE/VDE/cULus Electric Fuse, 
    Very Fast Blow, 8A, 250VAC, 1500A (IR), Inline/holder, 5x20mm': 8, 'Fuse Miniature Fast Acting 12.5A 250V Holder 
    Cartridge 5 X 20mm Ceramic Box PSE/cULus Electric Fuse, Very Fast Blow, 12.5A, 250VAC, 500A (IR), Inline/holder, 
    5x20mm': 13, 'Fuse Miniature Fast Acting 12.5A 250V Holder Cartridge 5 X 20mm Ceramic Bulk PSE/cULus Electric 
    Fuse, Very Fast Blow, 12.5A, 250VAC, 500A (IR), Inline/holder, 5x20mm': 16, 'Fuse Miniature Fast Acting 3.15A 
    250V Holder Cartridge 5 X 20mm Glass CCC/PSE/VDE/cULus Electric Fuse, Fast Blow, 3.15A, 250VAC, 35A (IR), 
    Inline/holder, 5x20mm': 24, 'Fuse Miniature Fast Acting 12.5A 250V Holder Cartridge 5 X 20mm Ceramic Bulk 
    CE/CSA/PSE/UL/VDE Electric Fuse, Fast Blow, 12.5A, 250VAC, 1500A (IR), Inline/holder, 5x20mm': 23} 
    .
    .
    .
    '''

    return res


if __name__ == '__main__':
    df = pd.read_csv("Fuse.csv", sep=";")
    pd.set_option('display.max_columns', None)

    # Do descriptive analysis (1° step)
    descriptive_analysis(df)

    df = df[['PART_DESCRIPTION']]
    # dropping duplicates on "PART_DESCRIPTION" column
    df.drop_duplicates(inplace=True)

    # Material similarity based on PART_DESCRIPTION using Levenshtein (2° step)
    # no need to do stopwords and punctuation removal with our dataset
    '''
    Informally, the Levenshtein distance between two words is the minimum number of single-character edits 
    (insertions, deletions or substitutions) required to change one word into the other.
    - https://en.wikipedia.org/wiki/Levenshtein_distance
    '''
    similar_materials = material_similarity(df)

    # How to use other feature for material similarity task? (3° steps)
    '''One way to take advantage of the other features for the similarity task could be to carry out the similarity 
    between the other features as already done with the PART_DESCRIPTION variable. That is to use a distance metric (
    Levenshtein in our case) also between rows of other columns. The goal will be to find that material that 
    minimizes the sum of the Levenshtein distances between the various columns of the same row. 

    As already specified in the first step, about 37% of values are missing. This is absolutely no big deal. In some 
    PART_DESCRIPTION there are technical information not present in the other features. In other cases, however, 
    this is not the case and the information is redundant. The missing values in the features could be filled from 
    the information present in the PART_DESCRIPTION column. Then we proceed with the minimization of the sum of the 
    distances on each row. 
    '''
