# tool macros
CC := nvcc

CCFLAGS :=
LDFLAGS := 
OBJFLAGS := -c

INC_DIRS=./inc 
LIBS= 
LIB_DIRS=

CCFLAGS+= $(foreach D,$(INC_DIRS),-I$D)
CCDBGFLAGS+= $(foreach D,$(INC_DIRS),-I$D)
LDFLAGS+= $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS), -l$(L))

# path macros
BIN_PATH := bin
OBJ_PATH := obj
SRC_PATH := src

# src files & obj files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# clean files list
DISTCLEAN_LIST := $(OBJ)
CLEAN_LIST := $(TARGET) \
			  $(DISTCLEAN_LIST)

# default rule
default: makedir all

# non-phony targets

$(BIN_PATH)/matmul-naive.app: $(OBJ_PATH)/matmul-naive.o
	$(CC) $< $(CCFLAGS) $(LDFLAGS) -o $@

$(BIN_PATH)/matmul-cublas.app: $(OBJ_PATH)/matmul-cublas.o
	$(CC) $< $(CCFLAGS) $(LDFLAGS) -o $@

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(CC) $< $(CCFLAGS) $(OBJFLAGS) -o $@

# phony rules

.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

.PHONY: all
all: $(OBJ) matmul

.PHONY: matmul
matmul: $(BIN_PATH)/matmul-naive.app $(BIN_PATH)/matmul-cublas.app

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(DISTCLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)