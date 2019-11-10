
import json


old_scenario=open("scenarios/New_scenario.scenario", "r")


# print(content_old_scenario)

new_scenario = open("scenarios/Modified_New_scenario.scenario", "w+")
i=0

dynamicElements = {
"source" : None,
"targetIds" : [ 2 ],
"position" : {
"x" : 1.5,
"y" : 1.8
},
"velocity" : {
"x" : 0.0,
"y" : 0.0
},
"nextTargetListIndex" : 0,
"freeFlowSpeed" : 1.2399599214195676,
"attributes" : {
"id" : -1,
"radius" : 0.2,
"densityDependentSpeed" : False,
"speedDistributionMean" : 1.34,
"speedDistributionStandardDeviation" : 0.26,
"minimumSpeed" : 0.5,
"maximumSpeed" : 2.2,
"acceleration" : 2.0,
"footStepsToStore" : 4,
"searchRadius" : 1.0,
"angleCalculationType" : "USE_CENTER",
"targetOrientationAngleThreshold" : 45.0
},
"idAsTarget" : -1,
"isChild" : False,
"isLikelyInjured" : False,
"mostImportantEvent" : None,
"salientBehavior" : "TARGET_ORIENTED",
"groupIds" : [ ],
"trajectory" : {
"footSteps" : [ ]
},
"groupSizes" : [ ],
"modelPedestrianMap" : None,
"type" : "PEDESTRIAN"
}

data = json.load(old_scenario)
print(data["scenario"]["topography"]["dynamicElements"])

data["scenario"]["topography"]["dynamicElements"] = json.dumps(dynamicElements)
# print(data["scenario"]["topography"]["dynamicElements"])

data["name"]= json.dumps("Modified_New_scenario.scenario")

json.dump(data, new_scenario, indent=4)


new_scenario.close()
old_scenario.close()
