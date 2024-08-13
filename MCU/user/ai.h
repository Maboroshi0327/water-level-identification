#ifndef _AI_H
#define _AI_H

#include "model_data.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ai_setup(void);
void ai_predict(float*, int, float*);

#ifdef __cplusplus
}
#endif

#endif /* _AI_H */
