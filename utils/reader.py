import pandas as pd


def parse_sequence_string(sequence_str, label=0):
    sequence_str = str(sequence_str).strip()
    if not sequence_str:
        return [str(label), set()]

    if '-1' in sequence_str:
        line_split = sequence_str.split('-1')
        split_first_itemset = line_split[0].split()
        if len(split_first_itemset) > 1:
            first_itemset = set(split_first_itemset[1:])
        elif split_first_itemset:
            first_itemset = {split_first_itemset[0]}
        else:
            first_itemset = set()

        sequence = [str(label), first_itemset]
        for itemset in line_split[1:]:
            items = itemset.replace('-2', '').strip().split()
            if items:
                sequence.append(set(items))
        return sequence

    tokens = [token for token in sequence_str.split() if token != '-2']
    return [str(label)] + [{token} for token in tokens]


def read_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    required_columns = {'sequence', 'y_true', 'confidence'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV {csv_path} must contain columns {sorted(required_columns)}; missing {sorted(missing)}"
        )

    data = [
        parse_sequence_string(row['sequence'], row['y_true'])
        for _, row in df.iterrows()
    ]
    target_class = df[['y_true', 'confidence']].values
    return data, target_class
