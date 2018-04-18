import csv
import numpy


input_file = csv.DictReader(open('data.csv'))
all_data = list(input_file)

aWeight= 0
countWeight= 0
valuesWeight = []
meanWeight = 0
results = {}
discrete_probabilities = {}

def CaluluateMeanSTDContinous(ParmeterName, rows):
    count = 0
    a = 0
    values_yes = []
    values_no = []
    for line in rows:
        if line["GP_greater_than_0"] == 'yes':
            weightstr = line[ParmeterName]
            a += float(weightstr)
            values_yes.append(float(weightstr))
            count += 1
        else:
            weightstr = line[ParmeterName]
            a += float(weightstr)
            values_no.append(float(weightstr))
            count += 1
    N_yes=len(values_yes)
    N_no = len(values_no)
    results[ParmeterName] = {
            'yes_mean': numpy.mean(values_yes),
            'yes_var': numpy.var(values_yes)*(N_yes/(N_yes-1)),
            'no_mean': numpy.mean(values_no),
            'no_var': numpy.var(values_no)*(N_no/(N_no-1))
        }

def CaluluatePriorProbability(rows, class_value):
    count = len(rows)
    count_class_value = 0
    for line in rows:
        if line["GP_greater_than_0"] == class_value:
            count_class_value += 1
    return float(count_class_value)/count

def CaluluateDiscreteProbability(ParmeterName, rows):
    param_prob_attr = {}
    all_param_values_discrete = []

    for line in rows:
        param_val = line[ParmeterName]
        if param_val not in param_prob_attr:
            all_param_values_discrete.append(param_val)
            param_prob_attr[param_val] = {'yes': 0, 'no': 0, 'count':0}

        param_prob_attr[param_val]['count'] +=1
        if line["GP_greater_than_0"] == 'yes':
            param_prob_attr[param_val]['yes'] += 1
        else:
            param_prob_attr[param_val]['no'] += 1

    discrete_probabilities[ParmeterName] = param_prob_attr

    for (key,value) in param_prob_attr.items():
        param_prob_attr[key]['yes'] = float(param_prob_attr[key]['yes'])/param_prob_attr[key]['count']
        param_prob_attr[key]['no'] = float(param_prob_attr[key]['no']) / param_prob_attr[key]['count']



def calculateConitinousProbability(parameter_name, parameter_value, class_value):
    mean = results[parameter_name][class_value + '_mean']
    var = results[parameter_name][class_value + '_var']
    x = float(parameter_value)
    pi = numpy.pi
    bb  = numpy.power(2*pi*var, 0.5)
    exp = numpy.power(numpy.e, (-1*numpy.power(x-mean,2)/(2*(var))))

    prob = exp /bb
    return prob


def split(filename):
    input_file = csv.DictReader(open(filename))
    rows = list(input_file)
    testrows = []
    trainrows = []
    for line in rows:
        year = int(line["DraftYear"])
        if year > 1997 and year < 2001:
            trainrows.append(line)
        if year == 2001:
            testrows.append(line)
    return trainrows, testrows





def testAccuracy(testdata):
    pass


train, test = split('data.csv')
testAccuracy(test)
continous_params = [
    'CSS_rank', 'DraftAge', 'Height', 'Overall', 'Weight', 'po_A', 'po_G', 'po_GP', 'po_P', 'po_PIM', 'rs_A', 'rs_G', 'rs_GP', 'rs_P', 'rs_PIM', 'rs_PlusMinus']

discrete_params = ['country_group', 'Position']
for p in continous_params:
    CaluluateMeanSTDContinous(p, train)

for p in discrete_params:
    CaluluateDiscreteProbability(p, train)





def naiveClassifier(rows):
    total_rows = len(rows)
    accuracy = 0
    for line in rows:
        result_yes = CaluluatePriorProbability(rows, 'yes')
        result_no = CaluluatePriorProbability(rows, 'no')
        for p in continous_params:
            param_val = line[p]
            result_yes =  result_yes * calculateConitinousProbability(p, param_val, 'yes')
            result_no = result_no * calculateConitinousProbability(p, param_val, 'no')
        for p1 in discrete_params:
            param_val = line[p1]
            result_yes = result_yes * discrete_probabilities[p1][param_val]['yes']
            result_no = result_no * discrete_probabilities[p1][param_val]['no']

        if float(result_yes) > float(result_no) and (line['GP_greater_than_0'] == 'yes'):
            accuracy += 1
        elif result_no > result_yes and (line['GP_greater_than_0'] == 'no'):
            accuracy += 1

    return (float(accuracy)/total_rows)*100

print ("The Accuracy Percentage using Bessels formula is: ")
print(naiveClassifier(test))
