PROG1	= $(shell jq -r '.acapPackageConf.setup.appName' manifest.json)
OBJS1	= $(PROG1).c audio.c detector.c fusion.c model.c wav.c web.c
PROGS	= $(PROG1)
DEBUG_DIR = debug

PKGS = glib-2.0 gio-2.0 axevent axparameter liblarod libpipewire-0.3 fcgi jansson

CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS)) -lm

CFLAGS += -O2 \
          -Wall \
          -Wextra \
          -Wformat=2 \
          -Wpointer-arith \
          -Wstrict-prototypes \
          -Wmissing-prototypes \
          -Wdisabled-optimization \
          -Wfloat-equal \
          -W \
          -Werror

all: $(PROGS)

$(PROG1): $(OBJS1)
	install -d $(DEBUG_DIR)
	$(CC) $^ $(CFLAGS) $(LIBS) $(LDFLAGS) $(LDLIBS) -o $(DEBUG_DIR)/$@
	cp $(DEBUG_DIR)/$@ .
	$(STRIP) $@

clean:
	rm -rf $(PROGS) *.o *.eap* *_LICENSE.txt package.conf* param.conf tmp* $(DEBUG_DIR)
