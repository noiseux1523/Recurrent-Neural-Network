#!/bin/bash

#
# Script to remove duplicates in respective train and test sets
#
# bash find-duplicates.sh /Users/noiseux1523/cnn-text-classification-tf-w2v/data/10-Fold
#

# System folder with folds
folder=$1;

# Verification
if [ -f $folder/dup.txt ] ; then
  rm $folder/dup.txt;
fi;

if [ -f $folder/dup-v2.txt ] ; then
  rm $folder/dup-v2.txt;
fi;

# Sytems to remove duplicates
cat ./system-dup.txt | while read sys;
do
	cd $folder/$sys;
	echo "DUPLICATES - $sys" >> $folder/dup.txt;

	# Check the 10 folds
	for i in {0..9}
	do
		# Get the file names
		train_positive="$i-train-$sys-bodies-yes.csv.pos";
		test_positive="$i-test-$sys-bodies-yes.csv.pos";
		train_negative="$i-train-$sys-bodies-no.csv.neg";
		test_negative="$i-test-$sys-bodies-no.csv.neg";

		# Copy files to keep a backup of original files
		cp test_positive "$i-test-$sys-bodies-yes-no-dup.csv.pos";
		cp test_negative "$i-test-$sys-bodies-no-no-dup.csv.neg";

		# Sort the training/test files, remove duplicates and print preprocessed files in new file
		sort $train_positive | uniq -u > train-pos.txt; 
		sort $test_positive | uniq -u > test-pos.txt; 
		sort $train_negative | uniq -u > train-neg.txt; 
		sort $test_negative | uniq -u > test-neg.txt; 

		# Concatenate positive/negative files, sort and print duplicates in log file
		echo "$train_positive + $test_positive" >> $folder/dup.txt;
		cat train-pos.txt test-pos.txt | sort | uniq -d >> $folder/dup.txt;
		echo "$train_negative + $test_negative" >> $folder/dup.txt;
		cat train-neg.txt test-neg.txt | sort | uniq -d >> $folder/dup.txt;

		# Read file with positive duplicates and remove duplicates from the positive test file
		cat train-pos.txt test-pos.txt | sort | uniq -d > ./remove-pos.txt;
		cat ./remove-pos.txt | while read rem
		do
			line=1;
			cat $test_positive | while read test
			do
				# Remove method from test set if it's in training set
				if [ "$rem" = "$test" ]
				then 
					sed -i '' "${line}d" "$i-test-$sys-bodies-yes-no-dup.csv.pos"
				fi;
				line=$((line+1));
			done

		done

		# Read file with negative duplicates and remove duplicates from the negative test file
		cat train-neg.txt test-neg.txt | sort | uniq -d > ./remove-neg.txt;
		cat ./remove-neg.txt | while read rem
		do
			line=1;
			cat $test_negative | while read test
			do
				# Remove method from test set if it's in training set
				if [ "$rem" = "$test" ]
				then 
					sed -i '' "${line}d" "$i-test-$sys-bodies-no-no-dup.csv.neg"
				fi;
				line=$((line+1));
			done

		done

	done
	cd ../..;

done









