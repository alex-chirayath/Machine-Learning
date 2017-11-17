import shutil
import glob

def fileConcat(outfilename,pathInput):
	read_files = glob.glob(pathInput)
	with open(outfilename, "wb") as outfile:
		# for f in read_files:
		# 	with open(f, "rb") as infile:
		# 		print infile.read()
		for f in read_files:
			with open(f, "rb") as infile:
				outfile.write(infile.read())

def main():
	fileConcat('pos_true.csv','op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/**/*.txt')
	fileConcat('pos_fake.csv','op_spam_v1.4/positive_polarity/deceptive_from_MTurk/**/*.txt')
	fileConcat('neg_true.csv','op_spam_v1.4/negative_polarity/truthful_from_Web/**/*.txt')
	fileConcat('neg_fake.csv','op_spam_v1.4/negative_polarity/deceptive_from_MTurk/**/*.txt')
	
if __name__ == '__main__':
    main()
