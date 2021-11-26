# tool macros
CC := nvcc

CCFLAGS :=
LDFLAGS := 
CCDBGFLAGS := -O0 -g -Wall -Wfatal-errors
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
DBG_PATH := debug

# targets

# src files & obj files
SRC := $(foreach x, $(SRC_PATH), $(wildcard $(addprefix $(x)/*,.c*)))
OBJ := $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))
OBJ_DEBUG := $(addprefix $(DBG_PATH)/, $(addsuffix .o, $(notdir $(basename $(SRC)))))

# clean files list
DISTCLEAN_LIST := $(OBJ) \
                  $(OBJ_DEBUG)
CLEAN_LIST := $(TARGET) \
			  $(TARGET_DEBUG) \
			  $(DISTCLEAN_LIST)

# default rule
default: makedir all

# non-phony targets

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $< $(CCFLAGS) $(OBJFLAGS) -o $@

$(DBG_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CC) $< $(CCDBGFLAGS) $(OBJFLAGS) -o $@

# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

.PHONY: all
all: $(TARGET)

.PHONY: debug
debug: $(TARGET_DEBUG)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(DISTCLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)