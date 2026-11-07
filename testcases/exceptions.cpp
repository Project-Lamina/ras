#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>
struct G { G(){ std::printf("ctor\n"); } ~G(){ std::printf("dtor\n"); } };
static G g;
int main(){
  std::vector<int> v; for(int i=0;i<10;i++) v.push_back(i*i);
  std::string s="sum="; int t=0; for(int x:v) t+=x; s+=std::to_string(t);
  try { if(t>0) throw std::runtime_error("boom"); } catch(const std::exception&e){ s+=" caught:"; s+=e.what(); }
  std::printf("%s\n", s.c_str());
  return t%5;
}
