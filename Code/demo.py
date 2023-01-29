from flask.templating import render_template
from nltk import metrics
import os
import csv
import flask
import all_lesk

app = flask.Flask(__name__)

@app.route("/")
def show_form():
    return render_template("index.html")

@app.route("/disambiguate", methods=["POST"])
def disambiguate_request():
    if flask.request.method == "POST":
        if flask.request.form["context"] and flask.request.form["target"]:
            context = flask.request.form["context"]
            target = flask.request.form["target"]
            targetWords = []
            if "," in target:
                split_words = target.split(",")
                targetWords.extend(split_words)
            else:
                targetWords.append(target)
            
            all_bestAnswer = []
            all_answerUD = [] 
            all_answerWN = []

            all_predicted_answer = []
            all_alternate_answer = []

            for t in targetWords:
                # lesk_sense, max_overlaps, max_sim, best_ud_sense, max_ud_overlaps, max_ud_sim, best_wn_sense, max_wn_overlaps, max_wn_sim = all_lesk.lesk(context, t.strip())
                #lesk_sense, max_overlaps, max_sim, best_ud_sense, max_ud_overlaps, max_ud_sim, best_wn_sense, max_wn_overlaps, max_wn_sim = all_lesk.lesk_soft_cosine(context, t.strip())
                lesk_sense, max_overlaps, max_sim, best_ud_sense, max_ud_overlaps, max_ud_sim, best_wn_sense, max_wn_overlaps, max_wn_sim = all_lesk.lesk_avg_cosine(context, t.strip())
                if best_ud_sense and lesk_sense.word == best_ud_sense.word and lesk_sense.definition == best_ud_sense.definition and max_overlaps == max_ud_overlaps and max_sim == max_ud_sim:                    
                    all_predicted_answer.append((best_ud_sense, max_ud_overlaps, max_ud_sim, best_ud_sense.get_source().__name__))
                    if best_wn_sense: 
                        all_alternate_answer.append((best_wn_sense, max_wn_overlaps, max_wn_sim, best_wn_sense.get_source().__name__))
                    else:
                        all_alternate_answer.append((None, 0, 0, ""))

                elif best_wn_sense and lesk_sense.word == best_wn_sense.word and lesk_sense.definition == best_wn_sense.definition and max_overlaps == max_wn_overlaps and max_sim == max_wn_sim:
                    all_predicted_answer.append((best_wn_sense, max_wn_overlaps, max_wn_sim, best_wn_sense.get_source().__name__))
                    if best_ud_sense: 
                        all_alternate_answer.append((best_ud_sense, max_ud_overlaps, max_ud_sim, best_ud_sense.get_source().__name__))
                    else:
                        all_alternate_answer.append((None, 0, 0, ""))
                        
                else:
                    all_predicted_answer.append((None, 0, 0, ""))
                    all_alternate_answer.append((None, 0, 0, ""))


            return render_template("results.html", context=context, target=targetWords,\
                all_alternate_answer=all_alternate_answer, all_predicted_answer=all_predicted_answer, resultsNum=len(all_alternate_answer))
                
        else:
            errorMessages = ["You are missing variables, try again"]
            if not flask.request.form["context"]:
                errorMessages.append("You didn't enter a context, please enter one before trying to proceed")
            if not flask.request.form["target"]:
                errorMessages.append("You didn't enter any target words please enter one or more separated by a comma before proceeding")
            return render_template("index.html", errors=errorMessages)
    else:
        return render_template("index.html", errors="Wrong form method")

app.run(debug=True)

    # evaluate_UD_result(context, target, answerUD, maxOverlaps, answerWN)


# context="the house is lit, it's on fire"
# target="lit"
# disambiguate(context, target)