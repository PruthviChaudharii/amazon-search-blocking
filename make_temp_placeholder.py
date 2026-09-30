import csv

print("Creating temporary structurally valid placeholder matching file solely for validator execution...")
s1_path = "dataset/test/test_source1.tsv"
out_path = "output/temp_validation_matching_placeholder.tsv"

with open(s1_path, 'r', encoding='utf-8') as in_f, open(out_path, 'w', encoding='utf-8', newline='') as out_f:
    reader = csv.reader(in_f, delimiter='\t')
    next(reader)
    out_f.write("source1_entity_id\tmatched_entity_ids\n")
    for row in reader:
        out_f.write(f"{row[0].strip()}\t\n")

print(f"Created temporary placeholder: {out_path}")
