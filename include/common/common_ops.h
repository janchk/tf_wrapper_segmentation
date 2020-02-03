//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_COMMON_OPS_H
#define TF_WRAPPER_COMMON_OPS_H

#include "fs_handling.h"

namespace common_ops
{
    /// \brief
    /// \param filepath
    /// \return
    std::string extract_class(const std::string &filepath);

    template<typename Type>
    inline void delete_safe (Type * &ptr)
    {
        delete ptr;
      //ptr = (Type *)(uintptr_t(NULL) - 1);        /* We are not hiding our mistakes by zeroing the pointer */
        ptr = NULL;
    }

    template<typename Type>
    inline void deletearr_safe (Type * &ptr)
    {
        delete[] ptr;
      //ptr = (Type *)(uintptr_t(NULL) - 1);        /* We are not hiding our mistakes by zeroing the pointer */
        ptr = NULL;
    }
}

#endif //TF_WRAPPER_COMMON_OPS_H
