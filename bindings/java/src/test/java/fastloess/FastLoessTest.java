package fastloess;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import org.junit.jupiter.api.Test;

class FastLoessTest {

    @Test
    void reportsVersion() {
        assertNotNull(FastLoess.version());
        assertFalse(FastLoess.version().isBlank());
    }
}
