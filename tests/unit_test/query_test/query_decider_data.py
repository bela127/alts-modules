from alts.core.query.query_decider import QueryDecider
import alts.modules.query.query_decider as qdm

#List of query deciders
query_deciders = [
    qdm.AllQueryDecider,
    qdm.NoQueryDecider,
    qdm.ThresholdQueryDecider,
    qdm.TopKQueryDecider
]

#Decisiveness of query decider: Yes


#Pass Function
def passer_generator(qc: QueryDecider):
    if type(qc) == qdm.AllQueryDecider:
        ...