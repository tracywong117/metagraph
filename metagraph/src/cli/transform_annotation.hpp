#ifndef __TRANSFORM_ANNOTATION_HPP__
#define __TRANSFORM_ANNOTATION_HPP__

#include <string>
#include <vector>

namespace mtg {
namespace cli {

class Config;

int transform_annotation(Config *config);

int merge_annotation(Config *config);

int relax_multi_brwt(Config *config);

std::vector<std::vector<uint64_t>>
parse_linkage_matrix(const std::string &filename);

} // namespace cli
} // namespace mtg

#endif // __TRANSFORM_ANNOTATION_HPP__
