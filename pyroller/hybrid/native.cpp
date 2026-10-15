#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>
namespace py = pybind11;

// Known ordered text only. This kernel never discovers lyric identities.
py::dict forced_path(py::array_t<float,py::array::c_style|py::array::forcecast> input,
                     const std::vector<int>& tokens,int blank,double unknown_cost,
                     py::dict onset_bounds=py::dict(), py::dict entry_costs=py::dict(),
                     const std::vector<int>& end_bounds={}) {
    auto x=input.unchecked<2>();
    const int T=x.shape(0),V=x.shape(1),S=2*int(tokens.size())+1;
    if(T<1 || tokens.empty() || blank<0 || blank>=V || !std::isfinite(unknown_cost)
       || unknown_cost<0 || size_t(T)*size_t(S)>32000000)
        throw std::invalid_argument("invalid or over-budget forced alignment");
    for(int label:tokens)if(label<0 || label>=V || label==blank)
        throw std::invalid_argument("invalid forced label");
    const double neg=-std::numeric_limits<double>::infinity();
    int end_low=0,end_high=T;
    if(!end_bounds.empty()) {
        if(end_bounds.size()!=2 || end_bounds[0]<1 || end_bounds[1]>T || end_bounds[0]>end_bounds[1])
            throw std::invalid_argument("invalid final end bounds");
        end_low=end_bounds[0];end_high=end_bounds[1];
    }
    std::vector<double> previous(S,neg),current(S,neg);
    std::vector<uint8_t> back(size_t(T)*S,0);
    std::vector<int> lower(tokens.size(),0),upper(tokens.size(),T);
    std::vector<std::vector<double>> costs(tokens.size());
    for(auto item:entry_costs) {
        int label=py::cast<int>(item.first);
        auto values=py::cast<std::vector<double>>(item.second);
        if(label<0 || label>=int(tokens.size()) || int(values.size())!=T)
            throw std::invalid_argument("invalid entry costs");
        for(double v:values)if(!std::isfinite(v) || v<0)throw std::invalid_argument("invalid entry cost");
        costs[label]=std::move(values);
    }
    for(auto item:onset_bounds) {
        int label=py::cast<int>(item.first);
        auto range=py::cast<std::vector<int>>(item.second);
        if(label<0 || label>=int(tokens.size()) || range.size()!=2 || range[0]<0 || range[1]>T || range[0]>=range[1])
            throw std::invalid_argument("invalid onset bounds");
        lower[label]=range[0];upper[label]=range[1];
    }
    previous[0]=0;
    for(int t=0;t<T;++t) {
        double nonblank=neg;
        for(int k=0;k<V;++k) {
            if(!std::isfinite(x(t,k)))throw std::invalid_argument("nonfinite forced evidence");
            if(k!=blank)nonblank=std::max(nonblank,double(x(t,k)));
        }
        const double background=std::max(double(x(t,blank)),nonblank-unknown_cost);
        for(int s=0;s<S;++s) {
            double best=previous[s];uint8_t move=0;
            bool enter=!(s%2) || (t>=lower[s/2] && t<upper[s/2]);
            if(s==S-1)enter=t>=end_low && t<=end_high;
            double penalty=(s%2 && !costs[s/2].empty())?costs[s/2][t]:0.;
            if(enter && s>0 && previous[s-1]-penalty>best){best=previous[s-1]-penalty;move=1;}
            if(enter && s>1 && s%2 && tokens[s/2]!=tokens[s/2-1] && previous[s-2]-penalty>best){best=previous[s-2]-penalty;move=2;}
            double emission=(s==0 || s==S-1)?background:double(x(t,s%2?tokens[s/2]:blank));
            current[s]=best+emission;back[size_t(t)*S+s]=move;
            if(s==S-2 && t>=end_high)current[s]=neg;
        }
        previous.swap(current);
    }
    int state=previous[S-1]>previous[S-2]?S-1:S-2;
    if(!std::isfinite(previous[state]))throw std::runtime_error("no complete forced path");
    const double score=previous[state];
    std::vector<int> starts(tokens.size(),T),ends(tokens.size(),0);
    std::vector<double> emissions(tokens.size(),0);
    std::vector<int> counts(tokens.size(),0);
    for(int t=T-1;t>=0;--t) {
        if(state%2) {
            int i=state/2;starts[i]=std::min(starts[i],t);ends[i]=std::max(ends[i],t+1);
            emissions[i]+=x(t,tokens[i]);counts[i]++;
        }
        state-=back[size_t(t)*S+state];
    }
    for(size_t i=0;i<tokens.size();++i) {
        if(!counts[i])throw std::runtime_error("incomplete forced traceback");
        emissions[i]/=counts[i];
    }
    py::dict result;result["starts"]=starts;result["ends"]=ends;
    result["mean_token_logp"]=emissions;result["score"]=score;return result;
}

// Max-marginal over FIRST entry into selected label states. Every entry is
// scored with a complete prefix AND suffix on exactly the same evidence.
// These are Viterbi scores, not calibrated probabilities or log-sum posteriors.
py::array_t<double> onset_profiles(py::array_t<float,py::array::c_style|py::array::forcecast> input,
                                  const std::vector<int>& tokens,const std::vector<int>& probes,
                                  int blank,double unknown_cost) {
    auto x=input.unchecked<2>();
    const int T=x.shape(0),V=x.shape(1),S=2*int(tokens.size())+1;
    if(T<1 || tokens.empty() || blank<0 || blank>=V || !std::isfinite(unknown_cost) || unknown_cost<0
       || size_t(T)*size_t(S)>8000000 || size_t(T)*probes.size()>1000000)
        throw std::invalid_argument("invalid or over-budget onset profiles");
    for(int token:tokens)if(token<0 || token>=V || token==blank)throw std::invalid_argument("invalid label");
    for(int probe:probes)if(probe<0 || probe>=int(tokens.size()))throw std::invalid_argument("invalid probe");
    const double neg=-std::numeric_limits<double>::infinity();
    std::vector<double> background(T),forward(size_t(T)*S,neg),previous(S,neg),current(S,neg);
    for(int t=0;t<T;++t) {
        double nonblank=neg;
        for(int k=0;k<V;++k) {
            if(!std::isfinite(x(t,k)))throw std::invalid_argument("nonfinite evidence");
            if(k!=blank)nonblank=std::max(nonblank,double(x(t,k)));
        }
        background[t]=std::max(double(x(t,blank)),nonblank-unknown_cost);
    }
    auto emission=[&](int t,int s) {return (s==0 || s==S-1)?background[t]:double(x(t,s%2?tokens[s/2]:blank));};
    auto skip=[&](int s) {return s>1 && s%2 && tokens[s/2]!=tokens[s/2-1];};
    previous[0]=0;
    for(int t=0;t<T;++t) {
        for(int s=0;s<S;++s) {
            double value=previous[s];
            if(s>0)value=std::max(value,previous[s-1]);
            if(skip(s))value=std::max(value,previous[s-2]);
            current[s]=value+emission(t,s);forward[size_t(t)*S+s]=current[s];
        }
        previous.swap(current);
    }
    if(!std::isfinite(std::max(previous[S-1],previous[S-2])))throw std::runtime_error("no complete profile path");
    py::array_t<double> result({int(probes.size()),T});auto out=result.mutable_unchecked<2>();
    std::vector<double> suffix(S,neg),next(S,neg);suffix[S-1]=suffix[S-2]=0;
    for(int t=T-1;t>=0;--t) {
        auto prefix=[&](int s) {return t==0?(s==0?0.:neg):forward[size_t(t-1)*S+s];};
        for(size_t k=0;k<probes.size();++k) {
            int s=2*probes[k]+1;
            double entry=prefix(s-1);
            if(skip(s))entry=std::max(entry,prefix(s-2));
            out(k,t)=entry+emission(t,s)+suffix[s];
        }
        for(int s=0;s<S;++s) {
            double value=emission(t,s)+suffix[s];
            if(s+1<S)value=std::max(value,emission(t,s+1)+suffix[s+1]);
            if(s+2<S && skip(s+2))value=std::max(value,emission(t,s+2)+suffix[s+2]);
            next[s]=value;
        }
        suffix.swap(next);
    }
    return result;
}

// Sum over ALL complete paths on the same ordered character graph. These
// normalized model weights are not calibrated correctness probabilities.
py::array_t<double> onset_posteriors(py::array_t<float,py::array::c_style|py::array::forcecast> input,
                                    const std::vector<int>& tokens,const std::vector<int>& probes,
                                    int blank,double unknown_cost) {
    auto x=input.unchecked<2>();
    const int T=x.shape(0),V=x.shape(1),S=2*int(tokens.size())+1;
    if(T<1 || tokens.empty() || blank<0 || blank>=V || !std::isfinite(unknown_cost) || unknown_cost<0
       || size_t(T)*S>8000000 || size_t(T)*probes.size()>1000000)
        throw std::invalid_argument("invalid or over-budget onset posteriors");
    for(int token:tokens)if(token<0 || token>=V || token==blank)throw std::invalid_argument("invalid label");
    for(int probe:probes)if(probe<0 || probe>=int(tokens.size()))throw std::invalid_argument("invalid probe");
    const double neg=-std::numeric_limits<double>::infinity();
    auto add=[&](double a,double b) {if(a==neg)return b;if(b==neg)return a;
        return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));};
    std::vector<double> bg(T),forward(size_t(T)*S,neg),previous(S,neg),current(S,neg);
    for(int t=0;t<T;++t) {
        double nonblank=neg;
        for(int k=0;k<V;++k) {
            if(!std::isfinite(x(t,k)))throw std::invalid_argument("nonfinite evidence");
            if(k!=blank)nonblank=std::max(nonblank,double(x(t,k)));
        }
        bg[t]=std::max(double(x(t,blank)),nonblank-unknown_cost);
    }
    auto emission=[&](int t,int s) {return (s==0 || s==S-1)?bg[t]:double(x(t,s%2?tokens[s/2]:blank));};
    auto skip=[&](int s) {return s>1 && s%2 && tokens[s/2]!=tokens[s/2-1];};
    previous[0]=0;
    for(int t=0;t<T;++t) {
        for(int s=0;s<S;++s) {
            double value=previous[s];
            if(s>0)value=add(value,previous[s-1]);
            if(skip(s))value=add(value,previous[s-2]);
            current[s]=value+emission(t,s);forward[size_t(t)*S+s]=current[s];
        }
        previous.swap(current);
    }
    const double z=add(previous[S-1],previous[S-2]);
    if(!std::isfinite(z))throw std::runtime_error("no complete posterior path");
    py::array_t<double> result({int(probes.size()),T});auto out=result.mutable_unchecked<2>();
    std::vector<double> suffix(S,neg),next(S,neg);suffix[S-1]=suffix[S-2]=0;
    for(int t=T-1;t>=0;--t) {
        auto prefix=[&](int s) {return t==0?(s==0?0.:neg):forward[size_t(t-1)*S+s];};
        for(size_t k=0;k<probes.size();++k) {
            int s=2*probes[k]+1;double entry=prefix(s-1);
            if(skip(s))entry=add(entry,prefix(s-2));
            out(k,t)=std::exp(entry+emission(t,s)+suffix[s]-z);
        }
        for(int s=0;s<S;++s) {
            double value=emission(t,s)+suffix[s];
            if(s+1<S)value=add(value,emission(t,s+1)+suffix[s+1]);
            if(s+2<S && skip(s+2))value=add(value,emission(t,s+2)+suffix[s+2]);
            next[s]=value;
        }
        suffix.swap(next);
    }
    return result;
}

PYBIND11_MODULE(_alignment, m) {
    m.def("forced_path", &forced_path, py::arg("input"), py::arg("tokens"), py::arg("blank")=0,
          py::arg("unknown_cost")=6.0, py::arg("onset_bounds")=py::dict(),py::arg("entry_costs")=py::dict(),
          py::arg("end_bounds")=std::vector<int>{});
    m.def("onset_profiles", &onset_profiles, py::arg("input"), py::arg("tokens"), py::arg("probes"),
          py::arg("blank")=0, py::arg("unknown_cost")=6.0);
    m.def("onset_posteriors", &onset_posteriors, py::arg("input"), py::arg("tokens"), py::arg("probes"),
          py::arg("blank")=0, py::arg("unknown_cost")=6.0);
}
