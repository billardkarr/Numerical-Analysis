from __future__ import division
import numpy as np

# this script computes my grade in this course.


hw = [50, 84, 100, 95, 90, 96]
proj = [103]
exams = [112, 108]
final = [100]

letters = ['A', 'A-']
cutoffs = [90, 85]


def get_letter_grade(grade):
    for i in range(len(cutoffs)):
        if round(grade) >= cutoffs[i]:
            return letters[i]
    return 'F'


def mean(list):
    if len(list) == 0:
        return float('nan')
    else:
        return np.mean(list)

hw_mean = mean(hw + 2 * proj)
exam_mean = mean(exams)
final_mean = mean(final)
grade = mean(3 * [hw_mean] + 4 * [exam_mean] + 3 * final)
letter = get_letter_grade(grade)

print "homework average: %g" % hw_mean
print "exam average: %g" % exam_mean
print "final exam: %g" % final_mean
print "current grade: %g, %s" % (grade, letter)
